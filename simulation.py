import streamlit as st
import random
import numpy as np
import matplotlib as plt
import time

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Sample Variance Bias Simulation",
    layout="centered"
)

# -----------------------------
# Title & explanation
# -----------------------------
st.title("Sample Variance: Bias vs Unbiased Estimator")

st.markdown(
    """
This app simulates **repeated random sampling from a finite population** to
illustrate an important statistical concept:

> **Why dividing by _n − 1_ produces an unbiased estimator of variance,
while dividing by _n_ produces a biased one.**

We repeatedly draw samples of random size and compute variance using **two formulas**.
As the simulation runs, you can observe how each estimator behaves over time.
"""
)

st.markdown("---")

# -----------------------------
# Theory section
# -----------------------------
st.subheader("The two variance formulas")

st.latex(
    r"""
    s_n^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2
    """
)

st.markdown(
    """
- Dividing by **n**
- This is the *maximum likelihood estimator*
- It **systematically underestimates** the true population variance
"""
)

st.latex(
    r"""
    s_{n-1}^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
    """
)

st.markdown(
    """
- Dividing by **n − 1**
- Applies **Bessel’s correction**
- This estimator is **unbiased**, meaning its *expected value equals the true variance*
"""
)

st.markdown("---")

st.subheader("Why does the bias occur?")

st.markdown(
    """
When we estimate variance, we use the **sample mean** instead of the true population mean.
This makes the sample points appear *closer together* than they really are.

Dividing by **n − 1** corrects for this lost degree of freedom.
"""
)

# -----------------------------
# Population setup
# -----------------------------
population = list(range(1, 21))
true_variance = np.var(population)

st.markdown(
    f"""
**Population:** integers from 1 to 20  
**True population variance:** `{true_variance:.2f}`
"""
)

# -----------------------------
# Session state
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.var_n = []
    st.session_state.var_n1 = []

# -----------------------------
# Variance functions
# -----------------------------
def variance_n(x):
    m = np.mean(x)
    return np.mean((np.array(x) - m) ** 2)

def variance_n1(x):
    m = np.mean(x)
    return np.sum((np.array(x) - m) ** 2) / (len(x) - 1)

# -----------------------------
# Controls
# -----------------------------
st.subheader("Simulation control")

col1, col2 = st.columns(2)

with col1:
    if st.button("Start / Pause simulation"):
        st.session_state.running = not st.session_state.running

with col2:
    if st.button("Clear / Reset simulation"):
        st.session_state.running = False
        st.session_state.var_n = []
        st.session_state.var_n1 = []
        st.success("Simulation reset! Click 'Start / Pause' to run again.")

# -----------------------------
# Placeholder for plots
# -----------------------------
plot_placeholder = st.empty()

# -----------------------------
# Simulation update function
# -----------------------------
def plot_simulation():
    if len(st.session_state.var_n) == 0:
        return  # nothing to plot yet

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top row: scatter of variance
    axes[0, 0].scatter(range(len(st.session_state.var_n)), st.session_state.var_n, alpha=0.6)
    axes[0, 0].axhline(true_variance, color='red', linestyle='--')
    axes[0, 0].set_title("Variance (divide by n)")
    axes[0, 0].set_xlabel("Run")
    axes[0, 0].set_ylabel("Estimate")

    axes[0, 1].scatter(range(len(st.session_state.var_n1)), st.session_state.var_n1, alpha=0.6)
    axes[0, 1].axhline(true_variance, color='red', linestyle='--')
    axes[0, 1].set_title("Variance (divide by n−1)")
    axes[0, 1].set_xlabel("Run")

    # Bottom row: bias boxplots
    bias_n = [v - true_variance for v in st.session_state.var_n]
    bias_n1 = [v - true_variance for v in st.session_state.var_n1]

    axes[1, 0].boxplot(bias_n)
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].set_title("Bias (divide by n)")
    axes[1, 0].set_ylabel("Estimate − True Variance")

    axes[1, 1].boxplot(bias_n1)
    axes[1, 1].axhline(0, color='red', linestyle='--')
    axes[1, 1].set_title("Bias (divide by n−1)")

    plt.tight_layout()
    plot_placeholder.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Simulation loop (runs only when Start is pressed)
# -----------------------------
while st.session_state.running:
    n = random.randint(2, 10)
    sample = random.sample(population, n)
    st.session_state.var_n.append(variance_n(sample))
    st.session_state.var_n1.append(variance_n1(sample))
    plot_simulation()
    time.sleep(0.35)

# -----------------------------
# Always plot current state even if paused
# -----------------------------
plot_simulation()

# -----------------------------
# Final explanation
# -----------------------------
st.markdown("---")
st.subheader("What to observe")

st.markdown(
    """
- The **divide-by-n estimator** consistently clusters *below* the true variance  
- The **divide-by-(n − 1) estimator** fluctuates around the true variance  
- The **bias boxplots** clearly show median and spread differences  

This illustrates the idea of **bias**: an estimator is unbiased if its *expected value* equals the true parameter.
"""
)
