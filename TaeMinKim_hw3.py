import numpy
import math
import matplotlib.pyplot as plt

default_var = 0.25            
default_time_step = 1 / 20   

class DroneKF:
    """
    State: x = [ h, v ]^T
        h: altitude [m]
        v: vertical velocity [m/s]

    Model Matrices:
        A = [[1, dt],
             [0, 1]]
        B = [[dt^2 / (2m)],
             [dt      /  m]]
        d = [[-0.5 g dt^2],
             [ -g dt      ]]

    Process covariance from input noise:
        Q = B * Sigma_u * B^T,   Sigma_u = var of thrust noise [N^2]
    Measurement:
        z = C x + v_k,  C = [1, 0],   v_k ~ N(0, Sigma_z) with Sigma_z reported each step
    """

    def __init__(self, mass, g=9.81):
        self.m = mass
        self.g = g
        self.state = numpy.zeros((2, 1))
        self.state_cov = numpy.identity(2) * default_var
        self.predicted_state = numpy.zeros((2, 1))
        self.predicted_state_cov = numpy.identity(2) * default_var
        self.time = None

    def set_state(self, timestamp, altitude=0.0, velocity=0.0):
        self.time = timestamp
        self.state[0, 0] = altitude
        self.state[1, 0] = velocity
        self.state_cov = numpy.identity(2) * default_var

    def A_matrix(self, delta_t=default_time_step):
        A = numpy.array([[1.0, delta_t],
                         [0.0, 1.0]])
        return A

    def B_matrix(self, delta_t=default_time_step):
        B = numpy.array([[delta_t**2 / (2.0 * self.m)],
                         [delta_t      /  self.m]])
        return B

    def d_vector(self, delta_t=default_time_step):
        d = numpy.array([[-0.5 * self.g * delta_t**2],
                         [      - self.g * delta_t   ]])
        return d

    def C_matrix(self):
        C = numpy.array([[1.0, 0.0]])
        return C

    def _predict(self, timestamp, thrust_u,
                 var_u=default_var):
        """
        Predict using control (thrust) and its variance.
        thrust_u : control input in Newtons (scalar)
        var_u    : variance of thrust input (Sigma_u), N^2
        """
        if self.time is None:
            raise RuntimeError("uninitialized filter (call set_state first)")

        delta_t = timestamp - self.time
        self.time = timestamp

        A = self.A_matrix(delta_t)
        B = self.B_matrix(delta_t)
        d = self.d_vector(delta_t)

        u = numpy.array([[thrust_u]])                
        Sigma_u = numpy.array([[var_u]])             

        self.predicted_state = A @ self.state + B @ u + d
        self.predicted_state_cov = (
            A @ self.state_cov @ A.transpose() +
            B @ Sigma_u @ B.transpose()
        )

    def _measure(self, z_h, var_zh=default_var):
        """
        Measurement update with scalar z (altitude) and reported variance var_zh.
        """
        C = self.C_matrix()
        sigma_z = numpy.array([[var_zh]])             # (1x1)
        z = numpy.array([[z_h]])                      # (1x1)

        SigmaXCT = self.predicted_state_cov @ C.transpose()    # (2x1)
        CSigmaXCT = C @ SigmaXCT                                 # (1x1)
        K = SigmaXCT @ numpy.linalg.inv(CSigmaXCT + sigma_z)     # (2x1)

        innovation = z - C @ self.predicted_state                # (1x1)
        self.state = self.predicted_state + K @ innovation

        I = numpy.identity(2)
        self.state_cov = (I - K @ C) @ self.predicted_state_cov  # simple form (lecture style)

    def advance_filter(self, timestamp, thrust_u, z_h,
                       var_u=default_var, var_zh=default_var):
        self._predict(timestamp, thrust_u, var_u)
        self._measure(z_h, var_zh)

    # State Utilities in lecture -> in this case, don't need(because C is [1, 0])
    def get_estimate(self):
        z = self.C_matrix() @ self.state
        z_cov = self.C_matrix() @ self.state_cov @ self.C_matrix().transpose()
        return z, z_cov

    def get_state(self):
        return self.state, self.state_cov

def run_sim():
    # ----- given parameters -----
    m_true = 0.25
    fs = 20.0
    dt = 1.0 / fs
    T_final = 5.0
    steps = int(T_final / dt)
    g = 9.81

    thrust_nom = 2.7
    thrust_var = 0.25        
    thrust_std = math.sqrt(thrust_var)

    # measurement variance per-sample, uniform in [0.01, 0.5] m^2
    R_min, R_max = 0.01, 0.5

    rng = numpy.random.default_rng()

    # ----- true plant matrices -----
    def A_true(dt):
        return numpy.array([[1.0, dt],
                            [0.0, 1.0]])
    def B_true(dt):
        return numpy.array([[dt**2 / (2.0 * m_true)],
                            [dt      /  m_true]])
    def d_true(dt):
        return numpy.array([[-0.5 * g * dt**2],
                            [      - g * dt   ]])

    x_true = numpy.zeros((2,1))   # [h; v] initial 0

    # ----- filters -----
    kf_matched = DroneKF(m_true)            # (c) matched mass
    kf_mismatch = DroneKF(1.10 * m_true)    # (d) +10% mass

    t = 0.0
    kf_matched.set_state(t, altitude=0.0, velocity=0.0)
    kf_mismatch.set_state(t, altitude=0.0, velocity=0.0)

    # ----- logging -----
    times = numpy.zeros(steps+1)
    h_truth = numpy.zeros(steps+1)       # (a)
    z_meas  = numpy.zeros(steps+1)       # (b)
    h_kf_c  = numpy.zeros(steps+1)       # (c)
    h_kf_d  = numpy.zeros(steps+1)       # (d)

    # initial logs
    times[0] = 0.0
    h_truth[0] = x_true[0,0]
    z_meas[0] = x_true[0,0]  # just to start plot at 0
    h_kf_c[0] = kf_matched.state[0,0]
    h_kf_d[0] = kf_mismatch.state[0,0]

    # ----- simulate -----
    for k in range(1, steps+1):
        t = k * dt
        times[k] = t

        # true input with thrust noise
        u_true = thrust_nom + rng.normal(0.0, thrust_std)

        # propagate truth
        x_true = A_true(dt) @ x_true + B_true(dt) @ numpy.array([[u_true]]) + d_true(dt)

        # measurement with *reported* time-varying variance
        R_k = rng.uniform(R_min, R_max)
        z_k = x_true[0,0] + rng.normal(0.0, math.sqrt(R_k))

        # (c) matched model KF uses thrust_nom and knows Sigma_u = thrust_var
        kf_matched.advance_filter(t, thrust_nom, z_k, var_u=thrust_var, var_zh=R_k)

        # (d) mismatched mass (+10%); same thrust_nom and same reported variances
        kf_mismatch.advance_filter(t, thrust_nom, z_k, var_u=thrust_var, var_zh=R_k)

        # log
        h_truth[k] = x_true[0,0]
        z_meas[k] = z_k
        h_kf_c[k] = kf_matched.state[0,0]
        h_kf_d[k] = kf_mismatch.state[0,0]

    # ----- plots -----
    plt.figure(figsize=(9,5))
    plt.plot(times, h_truth, label="(a) Ground truth (stolen)")
    plt.plot(times, z_meas,  label="(b) Sensor (face value)")
    plt.plot(times, h_kf_c,  label="(c) KF (matched model)")
    plt.plot(times, h_kf_d,  label="(d) KF (+10% mass)")
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude h [m]")
    plt.title("Drone Liftoff — Altitude (20 Hz, 5 s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # errors
    err_b = z_meas - h_truth
    err_c = h_kf_c - h_truth
    err_d = h_kf_d - h_truth

    plt.figure(figsize=(9,5))
    plt.plot(times, err_b, label="Sensor - Truth")
    plt.plot(times, err_c, label="KF matched - Truth")
    plt.plot(times, err_d, label="KF +10% mass - Truth")
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude Error [m]")
    plt.title("Estimation Errors vs Ground Truth")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # quick RMSE printout
    # rmse_b = float(numpy.sqrt(numpy.mean(err_b**2)))
    # rmse_c = float(numpy.sqrt(numpy.mean(err_c**2)))
    # rmse_d = float(numpy.sqrt(numpy.mean(err_d**2)))
    # print(f"RMSE  Sensor:   {rmse_b:.4f} m")
    # print(f"RMSE  KF match: {rmse_c:.4f} m")
    # print(f"RMSE  KF +10%:  {rmse_d:.4f} m")

    plt.show()

if __name__ == "__main__":
    run_sim()
