#pragma once

#include "dims.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <random>
#include <limits>
#include <string>

#define BLOCK_SIZE   256
#define CHUNK        34816
#define SEED         1234

void mppi_set_ctrl_mode(int mode, double sia, double sig);
void mppi_set_proposal_sigma(const double* h_sigma_prop);   // sigma of the per-rollout constant offset (default = cost sigma)
void mppi_set_deltas(double uwb_delta, double ctrl_delta);
void mppi_set_uwb_model(int model, const double* mix5);   // 0 Tukey (published), 1 LOS/NLOS mixture
void mppi_set_floor_penalty(double weight);
void mppi_set_floor_hard(int hard);     // floor_hard:=1: z < 0 rejects the sample

class MPPISolver {
public:

    MPPISolver(int N, int T, double dt, double gamma,
               const double* h_anchor,
               const double* h_gravity,
               const double* h_sigma,
               const double* tag_offsets,
               int num_anchors,
               int num_tags,
               unsigned long seed = SEED);
    ~MPPISolver();

    void setDt(double dt);
    void setTdoa(const double* anchorB, int n_slots);   // dataset util: slot ai = anchor pair (anchor[ai], anchorB[ai])
    int  numAnchors() const { return num_anchors_; }
    int  numTags()    const { return num_tags_; }
    int  numRanges()  const { return num_tags_ * num_anchors_; }

    void solve(double* h_Uopt,
               double* h_xn,
               const double* h_x0,
               const double* h_U0,
               const double* h_ranges,
               const double* h_stds,
               const double* h_accgyr);

    double w_uwb_         {0.2};
    double w_acc_         {0.001};
    double w_gyr_         {0.015};

    // integrity score of the newest epoch from the last solve() (see kims_solver.cu)
    double integrityScore(double tau_z, const double* accgyr, double sigma_u_acc,
                          double sigma_u_gyr, double* frac_ok, double* min_chi2) const;

    double last_ess_      {0.0};   // effective sample size (sum w)^2 / sum w^2 of the last solve

    // ---- basis mode (input_space_v2.md) -------------------------------------------------
    // Sample theta = (delta_b[6], c[k][6], k < K) from a diagonal prior, expand the window
    // perturbation as delta_b + sum_k c_k phi_k(t) with an orthonormal DCT table, weight by the
    // UWB likelihood only (set w_acc_ = w_gyr_ = 0), optionally refit a Gaussian proposal once.
    void setBasis(int K, int iters, double ess_min);            // builds the phi table for the current T, dt
    void setBasisPrior(const double* h_prior_std);              // 6 + 6K standard deviations, per solve
    int  basisDim() const { return K_ > 0 ? 6 + 6 * K_ + (dv_on_ ? 3 : 0) : 0; }
    int    dv_on_ {0};             // 1: sample a head-velocity increment delta v (last 3 dims of theta)
    double dv_gain_ {0.0};         // fraction of E[delta v] applied to the head velocity per solve (1/T)
    void solveBasis(double* h_Uopt, double* h_xn,
                    const double* h_x0, const double* h_U0,
                    const double* h_ranges, const double* h_stds, const double* h_accgyr,
                    double* h_theta_mean, double* h_theta_var);  // posterior mean / diagonal variance of theta
    int    prior_in_weight_ {0};   // 1: log w += log p(theta) (MAP-like best-of-N); 0: exact IS (weights = likelihood)
    double last_ess1_ {0.0}, last_ess2_ {0.0};
    double last_logpred_ {0.0};    // predictive log-likelihood of the newest window epoch (basis mode, final pass)
    double last_logpred1_ {0.0};   // same from the first pass (prior proposal)
    // head-state sampling: dx0 ~ N(0, P) on (dp, dv, dtheta); after the solve the head is the
    // weighted mean of the samples' states one step later and last_P1_ their weighted covariance
    int    head_on_ {0};
    void   setHeadCov(const double* P9);
    void   setHeadSeed(unsigned long sd) { head_rng_.seed(sd); }
    double last_P1_[81] = {0};
    // head:=2 (sample-based Laplace): white IMU densities for the one-step Q
    // (acc x y z in m/s^2/sqrt(Hz), gyro x y z in rad/s/sqrt(Hz)); diagnostics of the last solve
    void   setHeadQ(const double* q6) { for (int k = 0; k < 6; ++k) head_q_[k] = q6[k]; }
    double last_A_[81] = {0};      // information of the leaving epoch about the new head (dp, dv, dtheta)
    double last_dmean_[9] = {0};   // weighted mean of the head offsets (the correction applied to the head)
    double last_fit_r2_ {-1.0};    // R^2 of the quadratic fit of the leaving epoch's cost (-1: no range)
    // ---- sequential proposal -------------------------------------------------
    // Sampled vector v = (head offset: dp dv dtheta [+ dba dbg], input basis coefficients c, 6K). Target: arrival
    // prior N(0, P_H) x input prior N(0, diag(sd_c^2)) x UWB likelihood. Proposal: defensive mixture of the previous
    // solve's weighted posterior moved one epoch (weight seq_alpha_) and the target prior. When the first pass has
    // ESS < ess_min * N the proposal is rebuilt by adaptive tempering (Gaussian refits at ESS N/2 per stage).
    int    head_dim_ {9};          // 9 or 15 (carried IMU bias as head state)
    double seq_alpha_ {0.9};
    double seq_inflate_ {1.2};
    int    seq_max_stages_ {20};
    int    seq_debug_ {0};
    int    seq_headfix_ {0};
    // seq_refl:=rho: a third proposal component, the carried posterior reflected through the anchor plane
    // (position and velocity normal components), with weight alpha*rho; the importance weights use the full mixture, so
    // the estimator stays exact and the other side of the plane is always represented
    double seq_refl_ {0.0};
    double plane_c_[3] = {0, 0, 0}, plane_n_[3] = {0, 0, 1}; int plane_set_ {0};
    void   setPlane(const double* c, const double* n) { for (int k = 0; k < 3; ++k) { plane_c_[k] = c[k]; plane_n_[k] = n[k]; } plane_set_ = 1; }
    int    skip_ui_ {0};
    double *d_prop_par_ {nullptr}, *d_prop_x_ {nullptr}, *d_prop_lp_ {nullptr};   // GPU proposal draw buffers           // basisPass: do not copy the N x T x 6 rollout inputs to the host (seq path)
    double dbg_shift_[64] = {0}, dbg_ratio_[64] = {0}, dbg_wmin_ {0}, dbg_wmax_ {0}, dbg_wshift_ {0}; long dbg_n_ {0};
    double seq_done_ {0.25};        // tempering stops at beta 1 once ESS >= seq_done_ * N
    double seq_shrink_k_ {1.0};     // shrinkage strength: rho = n_eff / (n_eff + k V)
    int    seq_ewma_ {0};
    long   ewma_n_ {0};           // average the carried covariance over the last T solves
    int    seq_newmass_ {1};        // add (input prior variance)/T to the carried covariance
    int    seq_shrink_carry_ {1};   // shrink the carried covariance toward the proposal that was used
    void   setHeadCovN(const double* P, int H) { head_dim_ = H; headP_.assign(P, P + H * H); }
    // true_time: per-step time steps from the epoch timestamps and a basis table computed on
    // the true epoch times (both set every solve by the node)
    void   setStepDts(const double* h_dts);
    void   setPhi(const double* h_phi);
    int    true_dt_ {0};
    void   setHeadQb(const double* qb6) { for (int k = 0; k < 6; ++k) head_qb_[k] = qb6[k]; }
    void   setProposal(const double* mu, const double* S, int valid) {
        const int V = (rb_on_ ? 0 : head_dim_) + 6 * K_; prop_valid_ = valid;
        if (valid) { prop_mu_.assign(mu, mu + V); prop_S_.assign(S, S + V * V); }
    }
    // rb:=1 (section 15): only the input coefficients are sampled; the head/bias offset is integrated analytically per
    // sample (positions exactly linear in dp, dv; attitude and bias to first order)
    int    rb_on_ {0};
    int    rb_verify_ {0};         // rb_verify:=1: Laplace as proposal only; exact rollout at the drawn offset gives the weights
    double prof_[10] = {0};          // solveRB timing sums (ms): draw, GPU pass, host RB math, -, total; [5] passes
    long   prof_n_ {0};
    int    rb_checked_ {0};
    int    bias_filt_ {0};
    double bias_gain_ {1.0};
    double head_gain_ {1.0};
    // lm:=1 (matched Levenberg-Marquardt baseline): the same target as solveSeq (window cost x arrival prior x input
    // prior over the same V = H + 6K vector), but its mode by LM with finite-difference gradient/Hessian from the same GPU
    // rollouts, and the Laplace Gaussian N(mode, H^-1) in place of the weighted samples; everything after (head, proposal,
    // arrival covariance, filtered head) is unchanged, so only "sampling vs optimisation" differs
    int    lm_on_ {0};
    int    lm_iters_ {8};          // outer LM iterations (one stencil pass each)
    int    rollout_fp32_ {0};      // 1: single-precision rollout kernel in basisPass
    double lm_h_ {0.02};           // finite-difference step as a fraction of the prior standard deviation per dimension
    int    pass_n_ {0};            // basisPass: number of rollouts to run (0 = N_)
    int    last_lm_iters_ {0};
    long   seq_dump_ {0};          // seq_dump:=n: at solve n (1-based) write the posterior samples, (dp_z, dv_z) cost landscapes and the LM path
    std::string seq_dump_file_;
    long   seq_strip_from_ {0}, seq_strip_to_ {0}, seq_strip_step_ {1};   // per-solve (dp_z, weight) strips appended to seq_dump_file_
    double last_lm_J_ {0.0};
    // rbias:=1: a common range bias b as the 10th head coordinate (measured range = distance + b + noise),
    // sampled with the head, carried at bias_gain_ per solve (rb_head_), random walk rb_q_ in the arrival covariance
    int    rbias_on_ {0};
    double rb_head_ {0.0}, rb_q_ {0.0}, last_drb_ {0.0};
    double* d_rbias_ {nullptr};
    double last_xT_[DIM_X] = {0};   // solveSeq: weighted-mean state at the newest window epoch (lag-0 estimate)
    int    flush_ {0};               // solveSeq: also return the weighted-mean position at every window epoch (flush_pos_, T x 3)
    std::vector<double> flush_pos_;
    int    head_filt_ {0};         // solveSeq: head = dead-reckoning prediction + Kalman update of the leaving epoch
    double head_shift_[9] = {0};   // last filtered shift (diagnostic)       // solveSeq: fraction of the smoothed head shift carried per solve (1 = smoothed head)       // solveSeq: fraction of the window bias posterior mean carried per solve (1/T)         // hbias with rb: bias mean from the leaving epoch only (filtering update)
    void   solveRB(double* h_xn, const double* h_x0, const double* h_U0,
                   const double* h_ranges, const double* h_stds, const double* h_accgyr);
    void   solveSeq(double* h_xn, const double* h_x0, const double* h_U0,
                    const double* h_ranges, const double* h_stds, const double* h_accgyr);
    std::vector<double> headP_, prop_mu_, prop_S_, last_PH_;   // last_PH_: arrival covariance for the next head
    int    prop_valid_ {0};
    double last_db_[6] = {0};      // posterior mean of the bias offset (head_dim_ 15)
    int    last_stages_ {0};
    void saveRng();                // common random numbers across calls
    void restoreRng();
    int    last_iters_ {0};
    double last_cost_uwb_ {0.0};
    double last_cost_acc_ {0.0};
    double last_cost_gyr_ {0.0};

private:

    int    N_, T_, num_anchors_, num_tags_;
    double dt_, gamma_;

    curandStatePhilox4_32_10_t* d_states_;
    double* d_anchor_;
    double* d_gravity_;

    cudaStream_t h2dS_, d2hS_, rngS_, rollS_;
    cudaEvent_t  rngDone_;

    double* d_x0_;
    double* d_U0_;
    double* d_ranges_;
    double* d_stds_;
    double* d_accgyr_;
    double* d_noise_;
    double* d_Ui_;
    double* d_cost_;
    double* d_cost_uwb_;
    double* d_cost_acc_;
    double* d_cost_gyr_;
    double* d_wi_;
    double* d_Uopt_;
    double* d_xn_;

    int len_U0_, len_ranges_, len_accgyr_;

    std::vector<double> h_cost_, h_Ui_, h_wi_, h_cost_last_;
    double* d_cost_last_;

    int    K_ {0}, D_ {0}, iters_ {1};
    double* d_dv_ {nullptr};       // N x 3 per-sample head-velocity increments
    double* d_cost_newest_ {nullptr};      // N robust UWB cost of the newest epoch
    curandStatePhilox4_32_10_t* d_states_bak_ {nullptr};
    double *d_dx0_ {nullptr}, *d_x1_ {nullptr};
    std::vector<double> h_dx0_, h_dz_, h_x1_;
    double* d_xT_ {nullptr};
    std::vector<double> h_xT_;
    double* d_xall_ {nullptr};
    std::vector<double> h_xall_;
    double head_q_[6] = {0};
    double head_qb_[6] = {0};
    std::vector<double> h_prior_std_, h_phi_, h_phi_prev_;
    double* d_dts_ {nullptr};
    double* d_rb_ {nullptr};
    std::vector<double> h_rb_;
    double head_L_[81] = {0};
    std::mt19937_64 head_rng_ {12345};
    std::normal_distribution<double> head_nrm_ {0.0, 1.0};
    std::vector<double> h_cost_newest_;
    double ess_min_ {0.1};
    double *d_phi_ {nullptr}, *d_theta_ {nullptr}, *d_logpq_ {nullptr};
    double *d_mu_ {nullptr}, *d_L_ {nullptr}, *d_prior_std_ {nullptr};
    std::vector<double> h_theta_, h_logpq_;
    void basisPass(int use_L, double* h_lw);
};
