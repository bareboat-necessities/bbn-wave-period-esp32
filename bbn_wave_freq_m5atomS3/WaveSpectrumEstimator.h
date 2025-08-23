#pragma once

#include <ArduinoEigenDense.h>
#include <array>
#include <vector>
#include <cmath>
#include <limits>

template<int Nfreq, int Nblock>
class WaveSpectrumEstimator {
public:
    using Vec = Eigen::Matrix<double, Nfreq, 1>;

    struct PMFitResult {
        double alpha;
        double fp;
        double cost;
    };

WaveSpectrumEstimator(double fs_raw_ = 240.0,
                      int decimFactor_ = 5,
                      int shift_samples_ = 224,
                      bool hannEnabled_ = true)
    : fs_raw(fs_raw_), decimFactor(decimFactor_), shift(shift_samples_), hannEnabled(hannEnabled_)
{
    // --- 1. Validate shift ---
    if (shift > Nblock) shift = Nblock;

    // --- 2. Compute decimated sample rate ---
    fs = fs_raw / decimFactor;

    // --- 3. Initialize default frequency grid ---
    buildFrequencyGrid();

    // --- 4. Precompute Goertzel coefficients ---
    for (int i = 0; i < Nfreq; i++) {
        double k = 0.5 + (Nblock * freqs_[i]) / fs;   // mid-bin placement
        double omega = 2.0 * M_PI * k / Nblock;
        coeffs_[i] = 2.0 * std::cos(omega);
    }

    // --- 5. Initialize window ---
    for (int n = 0; n < Nblock; n++) {
        window_[n] = hannEnabled
                     ? 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (Nblock - 1)))
                     : 1.0;
    }
    windowGain = hannEnabled ? std::sqrt(3.0 / 8.0) : 1.0;  // RMS normalization

    // --- 6. Clear accumulators and buffers ---
    reset();

    // --- 7. Design low-pass biquad ---
    designLowpassBiquad(0.8);  // cutoff at 0.8 * Nyquist
}

    void reset(){
        buffer_.fill(0.0);
        s1_.setZero();
        s2_.setZero();
        s1_old_.setZero();
        s2_old_.setZero();
        writeIndex = 0;
        decimCounter = 0;
        samplesSinceLast = 0;
        z1 = z2 = 0.0;
        filledSamples = 0;
        isWarm = false;
    }

    void resetSpectrum(){
        s1_.setZero();
        s2_.setZero();
        s1_old_.setZero();
        s2_old_.setZero();
    }

bool processSample(double x_raw) {
    // --- 1. Apply low-pass biquad filter ---
    double y = b0 * x_raw + z1;
    z1 = b1 * x_raw + a1 * y + z2;
    z2 = b2 * x_raw + a2 * y;

    // --- 2. Decimation ---
    if (++decimCounter < decimFactor) return false;
    decimCounter = 0;

    // --- 3. Circular buffer: get oldest sample leaving the window ---
    int oldIdx = writeIndex;                  // oldest sample index
    double oldSample = buffer_[oldIdx];       // value leaving window

    // --- 4. Store new sample ---
    buffer_[oldIdx] = y;
    int newIdx = oldIdx;                       // index of new sample

    // --- 5. Increment write index ---
    writeIndex = (writeIndex + 1) % Nblock;

    // --- 6. Apply windowing ---
    double newWinSample = y * window_[newIdx];
    double oldWinSample = oldSample * window_[oldIdx];

    // --- 7. Update Goertzel accumulators ---
    for (int i = 0; i < Nfreq; i++) {
        // Current window contribution
        double s = newWinSample + coeffs_[i] * s1_[i] - s2_[i];
        s2_[i] = s1_[i]; s1_[i] = s;

        // Remove oldest sample contribution
        double so = oldWinSample + coeffs_[i] * s1_old_[i] - s2_old_[i];
        s2_old_[i] = s1_old_[i]; s1_old_[i] = so;
    }

    // --- 8. Warm-up check ---
    if (filledSamples < Nblock) {
        filledSamples++;
        if (filledSamples == Nblock) isWarm = true;
    }

    // --- 9. Shift / ready flag ---
    if (++samplesSinceLast >= shift && isWarm) {
        samplesSinceLast = 0;
        return true;
    }

    return false;
}

Vec getDisplacementSpectrum() const {
    Vec S;
    constexpr double g = 9.80665;

    for (int i = 0; i < Nfreq; i++) {
        // --- Numerically stable Goertzel magnitude ---
        double mag2_cur = (s1_[i] - coeffs_[i] * s2_[i]) * s1_[i] + s2_[i] * s2_[i];
        double mag2_old = (s1_old_[i] - coeffs_[i] * s2_old_[i]) * s1_old_[i] + s2_old_[i] * s2_old_[i];

        // --- Sliding window: subtract old sample contribution ---
        double mag2 = mag2_cur - mag2_old;

        // --- Normalize for window gain and block length ---
        mag2 /= (Nblock * Nblock * windowGain * windowGain);

        // --- Convert from acceleration spectrum to displacement spectrum ---
        double f = freqs_[i];
        S[i] = mag2 / std::pow(2.0 * M_PI * f, 4);

        // --- Safety: avoid negative due to numerical round-off ---
        if (S[i] < 0.0) S[i] = 0.0;
    }

    return S;
}

    double computeHs() const {
        Vec S = getDisplacementSpectrum();
        double m0 = 0.0;
        for(int i=0;i<Nfreq-1;i++){
            double df = freqs_[i+1]-freqs_[i];
            m0 += 0.5*(S[i]+S[i+1])*df;
        }
        return 4.0*std::sqrt(std::max(m0,0.0));
    }

    double estimateFp() const {
        Vec S = getDisplacementSpectrum();
        int idx=0; double maxVal=0;
        for(int i=0;i<Nfreq;i++){
            if(S[i]>maxVal){ maxVal=S[i]; idx=i;}
        }
        if(idx>0 && idx<Nfreq-1){
            double y0 = std::log(S[idx-1]);
            double y1 = std::log(S[idx]);
            double y2 = std::log(S[idx+1]);
            double p = 0.5*(y0 - y2)/(y0 - 2*y1 + y2);
            return freqs_[idx] + p*(freqs_[1]-freqs_[0]);
        }
        return freqs_[idx];
    }

    PMFitResult fitPiersonMoskowitz() const {
        auto S_obs = getDisplacementSpectrum();
        for(int i=0;i<Nfreq;i++) if(S_obs[i]<=0) S_obs[i]=1e-12;

        auto cost_fn = [&](double a,double fp){
            double omega_p=2*M_PI*fp; double cost=0.0;
            constexpr double beta=0.74;
            for(int i=0;i<Nfreq;i++){
                double model=a*9.80665*9.80665*std::pow(2*M_PI*freqs_[i],-5.0)
                    * std::exp(-beta*std::pow(omega_p/(2*M_PI*freqs_[i]),4.0));
                if(model<=0) model=1e-12;
                double d = std::log(S_obs[i])-std::log(model);
                cost += d*d;
            }
            return cost;
        };

        double bestA=1e-5,bestFp=0.1,bestC=std::numeric_limits<double>::infinity();
        for(int ia=0;ia<8;ia++){
            double a=1e-5 + ia*(1.0-1e-5)/7;
            for(int ifp=0;ifp<8;ifp++){
                double fp=0.03 + ifp*(1.0-0.03)/7;
                double c=cost_fn(a,fp);
                if(c<bestC){bestC=c;bestA=a;bestFp=fp;}
            }
        }

        double alpha=bestA, fp=bestFp, stepA=0.1, stepFp=0.1;
        for(int iter=0; iter<40; iter++){
            bool improved=false; double c;
            c=cost_fn(alpha+stepA, fp); if(c<bestC){bestC=c; alpha+=stepA; improved=true;}
            c=cost_fn(alpha-stepA, fp); if(c<bestC){bestC=c; alpha-=stepA; improved=true;}
            c=cost_fn(alpha, fp+stepFp); if(c<bestC){bestC=c; fp+=stepFp; improved=true;}
            c=cost_fn(alpha, fp-stepFp); if(c<bestC){bestC=c; fp-=stepFp; improved=true;}
            if(!improved){stepA*=0.5; stepFp*=0.5; if(stepA<1e-12 && stepFp<1e-12) break;}
        }

        PMFitResult res; res.alpha=alpha; res.fp=fp; res.cost=bestC;
        return res;
    }

    bool ready() const { return isWarm; }

private:
    void designLowpassBiquad(double normFc){
        double K = std::tan(M_PI*normFc);
        double norm = 1.0/(1.0+std::sqrt(2.0)*K + K*K);
        b0 = K*K*norm;
        b1 = 2*b0;
        b2 = b0;
        a1 = 2*(K*K-1)*norm;
        a2 = (1-sqrt(2.0)*K + K*K)*norm;
    }

    void buildFrequencyGrid() {
    constexpr double f_min = 0.03;  // Hz, lowest wave frequency
    constexpr double f_transition = 0.1; // Hz, where log → linear transition
    constexpr double f_max = 1.0;   // Hz, highest frequency of interest

    // Number of bins in log and linear portions
    int n_log = static_cast<int>(Nfreq * 0.4);   // ~40% of bins for log spacing
    int n_lin = Nfreq - n_log;

    // --- Logarithmic portion (f_min to f_transition) ---
    for (int i = 0; i < n_log; i++) {
        double t = static_cast<double>(i) / (n_log - 1);
        freqs_[i] = f_min * std::pow(f_transition / f_min, t);
    }

    // --- Linear portion (f_transition to f_max) ---
    for (int i = 0; i < n_lin; i++) {
        double t = static_cast<double>(i) / (n_lin - 1);
        freqs_[n_log + i] = f_transition + t * (f_max - f_transition);
    }
}

    double fs_raw, fs;
    int decimFactor, shift;
    bool hannEnabled;

    std::array<double,Nfreq> freqs_;
    std::array<double,Nfreq> coeffs_;
    std::array<double,Nblock> buffer_;
    std::array<double,Nblock> window_;
    double windowGain;

    // filter state
    double b0,b1,b2,a1,a2;
    double z1=0,z2=0;

    // Goertzel accumulators
    Eigen::Matrix<double,Nfreq,1> s1_, s2_, s1_old_, s2_old_;

    // counters
    int writeIndex=0;
    int decimCounter=0;
    int samplesSinceLast=0;

    // warm-up
    int filledSamples=0;
    bool isWarm=false;
};

