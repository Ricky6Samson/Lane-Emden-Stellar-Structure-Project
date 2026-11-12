
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Lane–Emden Stellar Structure",
    page_icon="🌌",
    layout="wide"
)

st.title("Numerical Modelling of Stellar Structure")
st.markdown(
    "### Exploring stellar density distributions using the Lane–Emden equation"
)

st.write(
    "An interactive numerical investigation of polytropic stellar models "
    "using the fourth-order Runge–Kutta method."
)

st.divider()


# --------------------------------------------------
# LANE-EMDEN SOLVER
# --------------------------------------------------

def lane_emden_rhs(xi, y, n):

    theta, u = y

    dtheta = u
    du = -(2 * u / xi) - max(theta, 0) ** n

    return np.array([dtheta, du])


def rk4_step(xi, y, h, n):

    k1 = lane_emden_rhs(xi, y, n)
    k2 = lane_emden_rhs(xi + h / 2, y + h * k1 / 2, n)
    k3 = lane_emden_rhs(xi + h / 2, y + h * k2 / 2, n)
    k4 = lane_emden_rhs(xi + h, y + h * k3, n)

    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)


@st.cache_data
def solve_lane_emden(n, h=0.005, max_xi=20):

    xi = 1e-4

    theta = 1 - xi**2 / 6 + n * xi**4 / 120
    u = -xi / 3 + n * xi**3 / 30

    y = np.array([theta, u])

    xis = [xi]
    thetas = [theta]
    us = [u]

    while xi < max_xi:

        step = min(h, max_xi - xi)

        y_next = rk4_step(xi, y, step, n)

        xi += step
        y = y_next

        xis.append(xi)
        thetas.append(y[0])
        us.append(y[1])

        if y[0] <= 0:
            break

    if thetas[-1] > 0:

        return (
            np.array(xis),
            np.array(thetas),
            np.array(us),
            None,
            None
        )

    xi_surface = xis[-2] - thetas[-2] * (
        (xis[-1] - xis[-2]) /
        (thetas[-1] - thetas[-2])
    )

    u_surface = us[-2] + (
        (us[-1] - us[-2]) *
        (xi_surface - xis[-2]) /
        (xis[-1] - xis[-2])
    )

    omega = -(xi_surface**2) * u_surface

    xi_values = np.array(xis[:-1] + [xi_surface])
    theta_values = np.array(thetas[:-1] + [0.0])
    u_values = np.array(us[:-1] + [u_surface])

    return xi_values, theta_values, u_values, xi_surface, omega


# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.title("Model Parameters")

n = st.sidebar.slider(
    "Polytropic index (n)",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.5
)

h = st.sidebar.select_slider(
    "Integration step size (h)",
    options=[0.02, 0.01, 0.005, 0.001],
    value=0.005
)

st.sidebar.divider()

st.sidebar.markdown("**Model assumptions**")
st.sidebar.caption(
    "Spherical symmetry, hydrostatic equilibrium, "
    "Newtonian gravity and a polytropic equation of state."
)


# --------------------------------------------------
# MODEL EXPLORER
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "Model Explorer",
    "Parameter Exploration",
    "Critical Case: n = 5"
])


with tab1:

    st.header("Stellar Structure Explorer")

    st.latex(r"P = K\rho^{1+1/n}")

    st.write(
        "The polytropic index determines the pressure-density "
        "relationship and influences the resulting stellar density profile."
    )

    xi, theta, u, xi_surface, omega = solve_lane_emden(
        n, h, 20
    )

    density = np.maximum(theta, 0) ** n

    enclosed_mass = -(xi**2) * u

    mass_fraction = enclosed_mass / omega

    concentration = xi_surface**3 / (3 * omega)

    col1, col2, col3 = st.columns(3)

    col1.metric("Surface Radius (ξ₁)", f"{xi_surface:.6f}")
    col2.metric("Mass Coefficient (ωₙ)", f"{omega:.6f}")
    col3.metric("Central / Mean Density", f"{concentration:.4f}")

    st.divider()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(xi, theta, linewidth=2)
    axes[0].set_title("Lane–Emden Solution")
    axes[0].set_xlabel("Dimensionless radius (ξ)")
    axes[0].set_ylabel("θ")
    axes[0].grid(alpha=0.3)

    axes[1].plot(xi, density, linewidth=2)
    axes[1].set_title("Normalised Density Profile")
    axes[1].set_xlabel("Dimensionless radius (ξ)")
    axes[1].set_ylabel("ρ / ρc")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Enclosed Mass Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        xi / xi_surface,
        mass_fraction,
        linewidth=2
    )

    ax.set_xlabel("Fractional radius (r / R)")
    ax.set_ylabel("Enclosed mass fraction (M(r) / M)")
    ax.set_title("Cumulative Mass Distribution")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# PARAMETER EXPLORATION
# --------------------------------------------------

with tab2:

    st.header("Effect of the Polytropic Index")

    st.write(
        "Investigate how changing the polytropic index "
        "affects the dimensionless properties of stellar models."
    )

    indices = [0.5, 1, 1.5, 2, 2.5, 3]

    results = []

    for index in indices:

        x, t, u_values, surface, mass = solve_lane_emden(
            index, h, 20
        )

        concentration = surface**3 / (3 * mass)

        results.append({
            "Polytropic Index": index,
            "Surface Radius": surface,
            "Mass Coefficient": mass,
            "Central / Mean Density": concentration
        })

    df = pd.DataFrame(results)

    st.dataframe(
        df.round(6),
        use_container_width=True,
        hide_index=True
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(
        df["Polytropic Index"],
        df["Surface Radius"],
        marker="o"
    )
    axes[0].set_title("Surface Radius")
    axes[0].set_xlabel("Polytropic index (n)")
    axes[0].set_ylabel("ξ₁")

    axes[1].plot(
        df["Polytropic Index"],
        df["Mass Coefficient"],
        marker="o"
    )
    axes[1].set_title("Mass Coefficient")
    axes[1].set_xlabel("Polytropic index (n)")
    axes[1].set_ylabel("ωₙ")

    axes[2].plot(
        df["Polytropic Index"],
        df["Central / Mean Density"],
        marker="o"
    )
    axes[2].set_title("Central Concentration")
    axes[2].set_xlabel("Polytropic index (n)")
    axes[2].set_ylabel("Central / Mean Density")

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Density Profile Comparison")

    fig, ax = plt.subplots(figsize=(9, 5))

    for index in [1.5, 3]:

        x, t, _, _, _ = solve_lane_emden(index, h, 20)

        ax.plot(
            x,
            np.maximum(t, 0)**index,
            label=f"n = {index}"
        )

    ax.set_xlabel("Dimensionless radius (ξ)")
    ax.set_ylabel("Normalised density (ρ / ρc)")
    ax.set_title("Density Profiles: n = 1.5 vs n = 3")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Enclosed Mass Comparison")

    fig, ax = plt.subplots(figsize=(9, 5))

    for index in [1.5, 3]:

        x, t, u_values, surface, mass = solve_lane_emden(
            index, h, 20
        )

        enclosed = -(x**2) * u_values
        fraction = enclosed / mass

        ax.plot(
            x / surface,
            fraction,
            label=f"n = {index}"
        )

    ax.set_xlabel("Fractional radius (r / R)")
    ax.set_ylabel("Enclosed mass fraction")
    ax.set_title("Mass Distribution Comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# CRITICAL POLYTROPIC INDEX: n = 5
# --------------------------------------------------

with tab3:

    st.header("The Critical Polytropic Index: n = 5")

    st.write(
        "The n = 5 Lane–Emden model has no finite stellar surface. "
        "Its density approaches zero asymptotically, while its total "
        "dimensionless mass approaches a finite limit."
    )

    st.latex(
        r"\theta(\xi) = \frac{1}{\sqrt{1+\xi^2/3}}"
    )

    xi5, theta5, u5, surface5, mass5 = solve_lane_emden(
        5, 0.005, 100
    )

    theta_exact = 1 / np.sqrt(1 + xi5**2 / 3)

    max_error = np.max(np.abs(theta5 - theta_exact))

    col1, col2 = st.columns(2)

    col1.metric("Finite Surface", "None")
    col2.metric("Maximum Absolute Error", f"{max_error:.3e}")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(xi5, theta5, label="Numerical RK4")
    ax.plot(
        xi5,
        theta_exact,
        "--",
        label="Analytical solution"
    )

    ax.set_xlabel("Dimensionless radius (ξ)")
    ax.set_ylabel("θ(ξ)")
    ax.set_title("Numerical vs Analytical Solution: n = 5")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Asymptotic Mass Convergence")

    numerical_mass = -(xi5**2) * u5
    analytical_mass = np.sqrt(3)

    limits = [5, 10, 20, 50, 100]

    convergence_results = []

    for radius in limits:

        idx = np.argmin(np.abs(xi5 - radius))

        mass_value = numerical_mass[idx]

        convergence_results.append({
            "Dimensionless Radius": radius,
            "Numerical Mass": mass_value,
            "Analytical Limit": analytical_mass,
            "Absolute Error": abs(
                analytical_mass - mass_value
            )
        })

    convergence_df = pd.DataFrame(convergence_results)

    st.dataframe(
        convergence_df.round(8),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        xi5,
        numerical_mass,
        label="Numerical enclosed mass"
    )

    ax.axhline(
        analytical_mass,
        linestyle="--",
        label="Analytical limit (√3)"
    )

    ax.set_xlabel("Dimensionless radius (ξ)")
    ax.set_ylabel("Dimensionless enclosed mass")
    ax.set_title("Asymptotic Mass Convergence: n = 5")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Numerical Modelling of Stellar Structure Using the Lane–Emden Equation"
)

st.caption(
    "Developed as an independent computational astrophysics project."
)
