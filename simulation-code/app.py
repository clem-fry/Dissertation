import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
import pandas as pd
import plots

import parameters
import analysis
import shapes
import simulation
import models
import time

# streamlit run app.py
# DISPERSION DIAGRAM
connectivity_dict = {'4': 1.1, '8': 1.9, '12': 2.1}
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Turing Patterns", "Initial Design", "Linear", "Nonlinear", "Postprocessing"])

def generate_params(dominant_turing_mode):
    valid_params = parameters.load_params('valid_params.txt')
    valid_params = [p[:5] for p in valid_params if p[-1] == True]

    desired_growth = 30 # exact
    desired_width = 0.1 # kind of

    optimal = parameters.find_suitable_params(dominant_turing_mode, desired_growth, desired_width, valid_params)
    option = 0
    a, b, c, d = optimal[option][0]

    #a, b, c, d = 49.373, -82.998, 61.178, -93.445
    sigma = optimal[option][1]
    Du = optimal[option][2]
    root_width = optimal[option][3]
    Dv = sigma * Du
    params = a, b, 0, c, d, 0, Du, Dv
    return params

@st.cache_data
def linear_pattern_data(t_array, lambda_alpha, eigvecs, grid_size):
    initial_conditions = np.random.rand(grid_size*grid_size, 2)
    initial_mode_inputs = np.dot(eigvecs.T, initial_conditions)

    u_history = []
    v_history = []

    for t in t_array:
        solution = eigvecs @ (np.exp(lambda_alpha * t)[:, None] * initial_mode_inputs)
        u, v = solution[:, 0].real, solution[:, 1].real

        u_history.append(u.copy())
        v_history.append(v.copy())

    return np.array(u_history), np.array(v_history)

@st.cache_data
def numerical_simulation_data(params, grid_size, threshold, n, random, lims):
    initial_conditions = np.random.rand(grid_size*grid_size, 2)
    a, b, x, c, d, x, Du, Dv = params
    f, g = models.linear(a, b, 0, c, d, 0)

    u_vector, v_vector, (u_log, v_log) = simulation.simulation_vectorised(f, g, Du, Dv, grid_size=grid_size, n = n, \
               dt = 0.001, show=False, save=False, initial_conditions=initial_conditions, threshold = threshold, limits=lims)
    
    return u_log, v_log


def make_heatmap(Z):
    fig = go.Figure(go.Heatmap(z=Z, showscale=False))

    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        height=400,
        uirevision="keep"
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(
        visible=False,
        scaleanchor="x",
        scaleratio=1
    )

    return fig

# DISPERSION DIAGRAM
grid_size = st.sidebar.slider("Grid Size", 5, 30, 30)

if "dominant_turing_mode" not in st.session_state:
    st.session_state.dominant_turing_mode = -0.5

if "last_desired_wavelength" not in st.session_state:
    st.session_state.last_desired_wavelength = grid_size / 4

dominant_turing_mode = st.sidebar.slider("Dominant Turing Mode", -10.0, -0.0, key="dominant_turing_mode")
connectivity = st.sidebar.radio("Connectivity", ['4', '8', '12'], key="connectivity")

params = generate_params(dominant_turing_mode)
lambda_alpha, eigvals = plots.dispersion_relation_plot(params, grid_size=grid_size, plot=False, threshold=connectivity_dict[connectivity])

def update_mode():
    wavelength = st.session_state.desired_wavelength
    connectivity = st.session_state.connectivity

    if connectivity == '4':
        S = -0.484
        R = 6.5
    elif connectivity == '8':
        S = -1.27
        R = 11.4
    else:
        S = -4.02
        R = 22.2

    suggested_mode = -( R / (wavelength - S) )

    st.session_state.dominant_turing_mode = float(
        np.clip(suggested_mode, -10.0, 0.0)
    )

with tab1:
    st.image("kilobot-update (1).png")

    st.latex(r"""
    \frac{d}{dt}u_i(t)
    =
    f(u_i,v_i)
    +
    \epsilon\sum_{j=1}^{N}L_{ij}u_j
    """)

    st.latex(r"""
    \frac{d}{dt}v_i(t)
    =
    g(u_i,v_i)
    +
    \sigma\epsilon\sum_{j=1}^{N}L_{ij}v_j
    """)

    st.latex(r"""
    \begin{aligned}
    f(u,v) &= Au - Bv \\
    g(u,v) &= Cu - Dv
    \end{aligned}
    """)

    _, centre, _ = st.columns([1, 2, 1])

    with centre:
        st.image("Giant_Pufferfish_skin_pattern_detail.jpg")

with tab2:

    # table = pd.DataFrame({
    #     "Connectivity": [4, 8, 12],
    #     "R": [6.5, 11.4, 22.2],
    #     "S": [-0.484, -1.27, -4.02]
    # })

    # st.table(table)

    desired_wavelength = st.number_input(
        "How many robots wide should the pattern features be?", min_value = 1.0, max_value = float(grid_size), value=grid_size/4, placeholder="What wavelength would you like?..."
    , key = "desired_wavelength", on_change=update_mode)

    if connectivity == '4':
        S = -0.484
        R = 6.5
    elif connectivity == '8':
        S = -1.27
        R = 11.4
    else: # connectivity = 12
        S = -4.02
        R = 22.2

    suggested_dominant_mode = -(R / (desired_wavelength - S))

    # if desired_wavelength != st.session_state.last_desired_wavelength:
    #     st.session_state.dominant_turing_mode = float(
    #         np.clip(suggested_dominant_mode, -10.0, 0.0)
    #     )
    #     st.session_state.last_desired_wavelength = desired_wavelength
    #     st.rerun()
        
    st.write("Dominant Turing Mode (hyperparameter) needed ", suggested_dominant_mode)

    a, b, x, c, d, x, Du, Dv = params

    st.write("Parameters used: ")
    st.write(f"A = {a}, B = {b}, C={c}, D={d}, Du={np.real(Du):.3f} and Dv={np.real(Dv):.3f}")

    st.image("wavelength-fit.png")

    st.image("coloured-connectivities.png")

    st.write("Relationship between dominant mode and wavelength:")
    st.latex(r"\ell = \frac{R}{\Lambda_m} + S")

with tab3:
    show_eigenvalues = st.toggle("Show eigenvalues")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=eigvals,
        y=lambda_alpha.real,
        mode="lines"
    ))

    if show_eigenvalues:
        fig.add_trace(go.Scatter(
            x=eigvals,
            y=lambda_alpha.real,
            mode="markers"
        ))

    fig.add_hline(y=0, line_dash="dash")

    fig.update_layout(
        xaxis_title="Laplacian Eigenvalue Λ",
        yaxis_title="Growth rate Re(λ)",
        uirevision="keep"
    )

    fig.update_xaxes(range=[-20, 0])
    fig.update_yaxes(range=[-20, 35])

    st.plotly_chart(fig, use_container_width=True, key='dispersion diagram')


    ###                      LINEAR TURING PATTERN PLOT

    if "animate_running" not in st.session_state:
        st.session_state.animate_running = False

    animate1 = st.button("▶ Run Animation")

    if animate1:
        st.session_state.animate_running = True

    # Set up linear equations:

    adjacency_matrix, degree_matrix, valid_nodes = simulation.generate_nodes(threshold=connectivity_dict[connectivity], grid_size=grid_size)
    L = adjacency_matrix - degree_matrix

    eigvals, eigvecs, lambda_alpha = analysis.linear_stability(params, grid_size = grid_size, threshold=connectivity_dict[connectivity])

    # Create plots:
    col1, col2 = st.columns(2)

    with col1:
        placeholder_u = st.empty()

    with col2:
        placeholder_v = st.empty()

    col3, col4 = st.columns(2)

    with col3:
        plot_line_plot2 = st.button("▶ Plot Lines", key = "plot lines linear")

    col5, col6 = st.columns(2)

    with col5:
        placeholder_line_u = st.empty()

    with col6:
        placeholder_line_v = st.empty()

    if "linear_line_plot" not in st.session_state:
        st.session_state.linear_line_plot = False

    if plot_line_plot2:
        st.session_state.linear_line_plot = True

    time_end = 2.0
    num_time_steps = 20

    t_slider = st.slider("Time", 0.0, time_end, time_end)

    times = np.linspace(0, time_end, num_time_steps)

    n_pixels = grid_size * grid_size
    u_analytic_history, v_analytic_history = linear_pattern_data(times, lambda_alpha, eigvecs, grid_size)

    if st.session_state.animate_running:
        for i in range(u_analytic_history.shape[0]):

            Z_u = u_analytic_history[i].reshape((grid_size, grid_size))
            Z_v = v_analytic_history[i].reshape((grid_size, grid_size))

            placeholder_u.plotly_chart(
                make_heatmap(Z_u),
                use_container_width=True,
                key = f'linear_u_heatmap_{i}'
            )

            placeholder_v.plotly_chart(
                make_heatmap(Z_v),
                use_container_width=True,
                key = f'linear_v_heatmap_{i}'
            )

            time.sleep(0.05)
        st.session_state.animate_running = False
    else:
        t_idx = np.searchsorted(times, t_slider)

        Z_u = u_analytic_history[t_idx].reshape((grid_size, grid_size))
        Z_v = v_analytic_history[t_idx].reshape((grid_size, grid_size))

        # Create Placeholders:
        placeholder_u.plotly_chart(make_heatmap(Z_u), use_container_width=True, config={"displayModeBar": False}, key = 'linear u heatmap timebar')
        placeholder_v.plotly_chart(make_heatmap(Z_v), use_container_width=True, config={"displayModeBar": False},  key = 'linear v heatmap timebar')


    # LINEAR LINE PLOTS
    if st.session_state.linear_line_plot:
        t_idx = np.searchsorted(times, t_slider)
        line_fig_u = go.Figure()
        line_fig_v = go.Figure()

        for pixel_idx in range(n_pixels):

            line_fig_u.add_trace(
                go.Scatter(
                    x=times[:t_idx],
                    y=u_analytic_history[:t_idx, pixel_idx],
                    mode="lines",
                    showlegend=False
                )
            )

            line_fig_v.add_trace(
                go.Scatter(
                    x=times[:t_idx],
                    y=v_analytic_history[:t_idx, pixel_idx],
                    mode="lines",
                    showlegend=False
                )
            )

        line_fig_u.update_xaxes(range=[0, time_end])
        line_fig_v.update_xaxes(range=[0, time_end])

        placeholder_line_u.plotly_chart(
            line_fig_u,
            use_container_width=True,
            config={"displayModeBar": False},
            key = 'linear line plot u'
        )

        placeholder_line_v.plotly_chart(
            line_fig_v,
            use_container_width=True,
            config={"displayModeBar": False},
            key = 'linear line plot v'
        )
    st.session_state.linear_line_plot = False


# ###                      NON LINEAR TURING PATTERN PLOT

with tab4:
    col1, col2 = st.columns(2)

    with col1:
        animate2 = st.button("▶ Run Animation", key = "run animation 2")

    with col2:
        new_initial_conditions = st.button("🔄 New Initial Conditions", key = "new initial condition")

    if "random_seed" not in st.session_state:
        st.session_state.random_seed = np.random.randint(1, 100)

    if new_initial_conditions:
        st.session_state.random_seed = np.random.randint(1, 100)

    if "animate_running" not in st.session_state:
        st.session_state.animate_running = False

    if animate2:
        st.session_state.animate_running = True

    # Create plots:
    col1, col2 = st.columns(2)

    with col1:
        placeholder_u2 = st.empty()

    with col2:
        placeholder_v2 = st.empty()

    col3, col4 = st.columns(2)

    with col3:
        plot_line_plot1 = st.button("▶ Plot Lines", key = "plot lines nonlinear")

    col5, col6 = st.columns(2)

    with col5:
        placeholder_line_u2 = st.empty()

    with col6:
        placeholder_line_v2 = st.empty()

    if "nonlinear_line_plot" not in st.session_state:
        st.session_state.nonlinear_line_plot = False

    if plot_line_plot1:
        st.session_state.nonlinear_line_plot = True

    num_time_steps = 5000
    log_interval = 1
    dt = 0.001
    time_end = dt * num_time_steps

    t_slider2 = st.slider("Time", 0.0, time_end, time_end, key='numeric time slider')
    
    times2 = np.arange(num_time_steps-1) * dt
    t_idx = np.searchsorted(times2, t_slider2)
    n_pixels = grid_size * grid_size

    u_lower_limit = st.number_input(
    "U lower limit", value=-65504, placeholder="Type a number...", key='u_lower_limit')
    
    u_upper_limit = st.number_input(
    "U upper limit", value=65504, placeholder="Type a number...", key='u_upper_limit')
    
    v_lower_limit = st.number_input(
    "V lower limit", value=-65504, placeholder="Type a number...", key='v_lower_limit')
    
    v_upper_limit = st.number_input(
    "V upper limit", value=65504, placeholder="Type a number...", key='v_upper_limit')

    limits = [u_lower_limit, u_upper_limit, v_lower_limit, v_upper_limit]
    
    u_numeric_history, v_numeric_history = numerical_simulation_data(params, grid_size, threshold=connectivity_dict[connectivity], n=num_time_steps, random=st.session_state.random_seed, lims = limits)

    # generate unique indicies weighted towards start for the animation
    p = 2
    x = np.linspace(0, 1, 150)

    indices = np.empty(150, dtype=int)
    indices[0] = 0

    for k in range(1, 150):
        target = round((num_time_steps - 1) * x[k]**p)
        indices[k] = max(target, indices[k-1] + 1)

    indices[-1] = num_time_steps - 1

    if st.session_state.animate_running:
        for i in indices:#np.linspace(0, num_time_steps - 1, 150, dtype=int):

            Z_u = u_numeric_history[:, i].reshape((grid_size, grid_size))
            Z_v = v_numeric_history[:, i].reshape((grid_size, grid_size))

            placeholder_u2.plotly_chart(
                make_heatmap(Z_u),
                use_container_width=True,
                key = f'nonlinear_u_heatmap_{i}'
            )

            placeholder_v2.plotly_chart(
                make_heatmap(Z_v),
                use_container_width=True,
                key = f'nonlinear_v_heatmap_{i}'
            )

            time.sleep(0.05)
        st.session_state.animate_running = False
    else:

        Z_u = u_numeric_history[:, t_idx].reshape((grid_size, grid_size))
        Z_v = v_numeric_history[:, t_idx].reshape((grid_size, grid_size))

        # Create Placeholders:
        placeholder_u2.plotly_chart(make_heatmap(Z_u), use_container_width=True, config={"displayModeBar": False}, key = 'nonlinear heatmap timebar u')
        placeholder_v2.plotly_chart(make_heatmap(Z_v), use_container_width=True, config={"displayModeBar": False}, key = 'nonlinear heatmap timebar v')


    # NONLINEAR LINE PLOTS
    t_idx2 = np.searchsorted(times2, t_slider2)
    line_fig_u2 = go.Figure()
    line_fig_v2 = go.Figure()

    if st.session_state.nonlinear_line_plot:
        for pixel_idx in range(n_pixels):

            line_fig_u2.add_trace(
                go.Scatter(
                    x=times2[:t_idx2],
                    y=u_numeric_history[pixel_idx, :t_idx2],
                    mode="lines",
                    showlegend=False
                )
            )

            line_fig_v2.add_trace(
                go.Scatter(
                    x=times2[:t_idx2],
                    y=v_numeric_history[pixel_idx, :t_idx2],
                    mode="lines",
                    showlegend=False
                )
            )

        line_fig_u2.update_xaxes(range=[0, time_end])
        line_fig_v2.update_xaxes(range=[0, time_end])

        placeholder_line_u2.plotly_chart(
            line_fig_u2,
            use_container_width=True,
            config={"displayModeBar": False},
            key = 'nonlinear line plot u'
        )

        placeholder_line_v2.plotly_chart(
            line_fig_v2,
            use_container_width=True,
            config={"displayModeBar": False},
            key = 'nonlinear line plot v'
        )
    st.session_state.nonlinear_line_plot = False

with tab5:
    col1, col2 = st.columns(2)

    with col1:
        plot_line_plot1 = st.button("▶ Analyse wavelength of the resulting pattern", key = "plot pattern analysis")

    if "analyse_results" not in st.session_state:
        st.session_state.analysis_running = False

    if plot_line_plot1:
        st.session_state.analysis_running = True

    col3, col4 = st.columns(2)

    with col3:
        spectral_analysis_u = st.empty()

    with col4:
        spectral_analysis_v = st.empty()

    if st.session_state.analysis_running:

        wavelengths_u, power_flat_u = analysis.plot_wavelength_spectrum(Z_u, plot_spectrum=False, log_scale=True, normalise=False)
        wavelengths_v, power_flat_v = analysis.plot_wavelength_spectrum(Z_v, plot_spectrum=False, log_scale=True, normalise=False)

        col3, col4 = st.columns(2)

        with col3:
            fig_u = go.Figure()

            fig_u.add_scatter(
                x=wavelengths_u,
                y=power_flat_u,
                mode="markers",
                marker_size=4,
            )

            fig_u.update_layout(
                title="U Spectrum",
                xaxis_title="Wavelength ℓ",
                yaxis_title="Power",
            )

            st.plotly_chart(fig_u, use_container_width=True)

        with col4:
            fig_v = go.Figure()

            fig_v.add_scatter(
                x=wavelengths_v,
                y=power_flat_v,
                mode="markers",
                marker_size=4,
            )

            fig_v.update_layout(
                title="V Spectrum",
                xaxis_title="Wavelength ℓ",
                yaxis_title="Power",
            )

            st.plotly_chart(fig_v, use_container_width=True)

        idx_u = np.argmax(power_flat_u)
        idx_v = np.argmax(power_flat_v)

        st.write(
            f"U dominant wavelength = {wavelengths_u[idx_u]:.2f} "
        )

        st.write(
            f"V dominant wavelength = {wavelengths_v[idx_v]:.2f} "
        )