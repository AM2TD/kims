#pragma once
// Matched optimisation baselines for the KIMS basis mode.
//
// Same window, same fixed head state, same parameters theta (DCT coefficients of the IMU input
// compensation, 6 channels x K modes), same prior N(0, diag(prior_std^2)), same robust range loss
// (Tukey biweight with cut delta in std units), same likelihood scaling gamma. The sampler's posterior is
//     p(theta | z) ∝ exp(-gamma * sum_r rho(r)) * N(theta; 0, S),
// and these baselines return its mode by minimising
//     J(theta) = gamma * sum_r rho(r(theta)) + 1/2 * theta' S^-1 theta
// with Gauss-Newton (IRLS weights for the Tukey loss, full steps) or Levenberg-Marquardt (Marquardt
// scaling, accept/reject). Jacobians by forward differences through the exact CPU copy of the rollout.
// Every window starts from theta = 0 (the sampler's proposal mean): no warm start, like the sampler.

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace kims {

struct BasisOptResult { int iters = 0; double J0 = 0, J = 0; bool ok = true; };

class BasisOpt {
public:
    // anchors: A x 3, tag offsets: G x 3 (body frame), gravity: 3
    BasisOpt(const std::vector<double>& anchors, int num_anchors,
             const std::vector<double>& tag_offsets, int num_tags, const double* gravity)
        : anc_(anchors), off_(tag_offsets), A_(num_anchors), G_(num_tags) {
        for (int i = 0; i < 3; ++i) g_[i] = gravity[i];
    }

    // phi: K x T (same table as the GPU), active: indices of theta that are optimised
    void setBasis(int K, int T, const std::vector<double>& phi) { K_ = K; T_ = T; phi_ = phi; }

    // mode 1: Gauss-Newton, 2: Levenberg-Marquardt
    BasisOptResult solve(int mode, int max_iter, double gamma, double delta,
                         const double* x0, const double* U0, double dt,
                         const double* ranges, const double* stds,
                         const std::vector<double>& prior_std, const std::vector<int>& active,
                         std::vector<double>& theta) const {
        const int D = static_cast<int>(prior_std.size());
        const int n = static_cast<int>(active.size());
        theta.assign(D, 0.0);
        BasisOptResult res;
        std::vector<double> r0, r1;
        double J = cost(theta, gamma, delta, x0, U0, dt, ranges, stds, prior_std, active, &r0);
        res.J0 = J;
        double lambda = 1e-3;
        for (int it = 0; it < max_iter; ++it) {
            // IRLS weights and FD Jacobian of the scaled residuals r_i (already divided by std)
            const int m = static_cast<int>(r0.size());
            Eigen::MatrixXd Jm(m, n);
            for (int c = 0; c < n; ++c) {
                const int d = active[c];
                const double eps = 1e-4 * prior_std[d];
                std::vector<double> th = theta; th[d] += eps;
                residuals(th, x0, U0, dt, ranges, stds, &r1);
                for (int i = 0; i < m; ++i) Jm(i, c) = (r1[i] - r0[i]) / eps;
            }
            Eigen::VectorXd w(m), r(m);
            for (int i = 0; i < m; ++i) {
                r(i) = r0[i];
                const double u = r0[i] / delta;
                w(i) = std::fabs(u) < 1.0 ? (1.0 - u * u) * (1.0 - u * u) : 0.0;
            }
            Eigen::MatrixXd H = gamma * Jm.transpose() * w.asDiagonal() * Jm;
            Eigen::VectorXd grad = gamma * Jm.transpose() * (w.array() * r.array()).matrix();
            for (int c = 0; c < n; ++c) {
                const int d = active[c];
                const double ip = 1.0 / (prior_std[d] * prior_std[d]);
                H(c, c) += ip;
                grad(c) += theta[d] * ip;
            }
            res.iters = it + 1;
            if (mode == 1) {                                        // Gauss-Newton: full step
                Eigen::VectorXd step = H.ldlt().solve(-grad);
                for (int c = 0; c < n; ++c) theta[active[c]] += step(c);
                J = cost(theta, gamma, delta, x0, U0, dt, ranges, stds, prior_std, active, &r0);
                if (!std::isfinite(J)) { res.ok = false; break; }
                if (step.norm() < 1e-9) break;
            } else {                                                // Levenberg-Marquardt
                bool accepted = false;
                for (int tries = 0; tries < 10 && !accepted; ++tries) {
                    Eigen::MatrixXd Hd = H;
                    for (int c = 0; c < n; ++c) Hd(c, c) += lambda * H(c, c);
                    Eigen::VectorXd step = Hd.ldlt().solve(-grad);
                    std::vector<double> th = theta;
                    for (int c = 0; c < n; ++c) th[active[c]] += step(c);
                    std::vector<double> rt;
                    const double Jt = cost(th, gamma, delta, x0, U0, dt, ranges, stds, prior_std, active, &rt);
                    if (std::isfinite(Jt) && Jt < J) {
                        const double rel = (J - Jt) / std::max(1e-12, J);
                        theta = th; J = Jt; r0 = rt; lambda = std::max(1e-9, lambda / 10.0); accepted = true;
                        if (rel < 1e-9) { res.J = J; return res; }
                    } else {
                        lambda *= 10.0;
                    }
                }
                if (!accepted) break;                               // no decrease possible: converged
            }
        }
        res.J = J;
        return res;
    }

    // one step of the corrected input from the head (same as step_dyn_kernel with U*_0)
    void stepHead(const double* x0, const double* u, double dt, double* xn) const { step(x0, u, dt, xn); }

private:
    std::vector<double> anc_, off_, phi_;
    int A_, G_, K_ = 0, T_ = 0;
    double g_[3];

    static void expSO3(const double* w, double* R) {   // exact copy of the device Exp_SO3 (row major)
        const double angle = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
        R[0] = 1; R[1] = 0; R[2] = 0; R[3] = 0; R[4] = 1; R[5] = 0; R[6] = 0; R[7] = 0; R[8] = 1;
        if (angle < 1e-6) return;
        const double a[3] = {w[0]/angle, w[1]/angle, w[2]/angle};
        const double c = std::cos(angle), s = std::sin(angle);
        const double Ah[9] = {0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0};
        double A2[9];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
            A2[3*i+j] = 0; for (int k = 0; k < 3; ++k) A2[3*i+j] += Ah[3*i+k] * Ah[3*k+j];
        }
        for (int i = 0; i < 9; ++i) R[i] += s * Ah[i] + (1.0 - c) * A2[i];
    }

    void step(const double* x, const double* u, double dt, double* xn) const {  // copy of step_dynamics
        double aw[3];
        aw[0] = x[6]*u[0]  + x[7]*u[1]  + x[8]*u[2]  + g_[0];
        aw[1] = x[9]*u[0]  + x[10]*u[1] + x[11]*u[2] + g_[1];
        aw[2] = x[12]*u[0] + x[13]*u[1] + x[14]*u[2] + g_[2];
        for (int i = 0; i < 3; ++i) {
            xn[i]     = x[i] + x[3+i]*dt + 0.5*aw[i]*dt*dt;
            xn[3+i]   = x[3+i] + aw[i]*dt;
        }
        const double om[3] = {u[3]*dt, u[4]*dt, u[5]*dt};
        double E[9]; expSO3(om, E);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                xn[6 + 3*r + c] = x[6 + 3*r + 0]*E[0*3 + c] + x[6 + 3*r + 1]*E[1*3 + c] + x[6 + 3*r + 2]*E[2*3 + c];
    }

    // scaled range residuals (d - z)/std over the window for valid ranges
    void residuals(const std::vector<double>& th, const double* x0, const double* U0, double dt,
                   const double* ranges, const double* stds, std::vector<double>* out) const {
        const int nr = G_ * A_;
        out->clear();
        double x[15], xn[15];
        for (int i = 0; i < 15; ++i) x[i] = x0[i];
        for (int t = 0; t < T_; ++t) {
            double u[6];
            for (int ch = 0; ch < 6; ++ch) {
                double v = U0[t*6 + ch] + th[ch];
                for (int k = 0; k < K_; ++k) v += th[6 + 6*k + ch] * phi_[k*T_ + t];
                u[ch] = v;
            }
            step(x, u, dt, xn);
            for (int gi = 0; gi < G_; ++gi) {
                double tw[3];
                for (int i = 0; i < 3; ++i)
                    tw[i] = xn[i] + xn[6+3*i]*off_[3*gi] + xn[7+3*i]*off_[3*gi+1] + xn[8+3*i]*off_[3*gi+2];
                for (int ai = 0; ai < A_; ++ai) {
                    const int ri = gi * A_ + ai;
                    const double z = ranges[t*nr + ri];
                    if (z <= 0.0) continue;
                    const double dx = tw[0]-anc_[3*ai], dy = tw[1]-anc_[3*ai+1], dz = tw[2]-anc_[3*ai+2];
                    out->push_back((std::sqrt(dx*dx + dy*dy + dz*dz) - z) / stds[t*nr + ri]);
                }
            }
            for (int i = 0; i < 15; ++i) x[i] = xn[i];
        }
    }

    double cost(const std::vector<double>& th, double gamma, double delta,
                const double* x0, const double* U0, double dt, const double* ranges, const double* stds,
                const std::vector<double>& prior_std, const std::vector<int>& active,
                std::vector<double>* r) const {
        residuals(th, x0, U0, dt, ranges, stds, r);
        double s = 0;
        for (double v : *r) {
            const double u = v / delta;
            s += std::fabs(u) >= 1.0 ? delta*delta/6.0 : delta*delta/6.0 * (1.0 - std::pow(1.0 - u*u, 3));
        }
        double p = 0;
        for (int d : active) p += 0.5 * (th[d] / prior_std[d]) * (th[d] / prior_std[d]);
        return gamma * s + p;
    }
};

}  // namespace kims
