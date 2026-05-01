import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- PAGE SETUP ---
st.set_page_config(page_title="Alpha Stability Optimizer", layout="wide")
st.title("Alpha System Stability Optimizer")

# --- INITIAL SYSTEM PARAMETERS ---
# Shifted +160.59mm to put rear wheels at X=0
m_stat_init = 145.08
com_x_stat = 319.59  
com_z_stat = 858.0
com_y_stat = 1.81
MIN_FS = 1.2

# --- SIDEBAR SLIDERS ---
st.sidebar.header("Adjust Parameters")

# 1. Initialize session state memory for our original defaults
defaults = {
    'cw_m': 63.38,
    'cw_z': 178.0,
    'cw_x': 319.59,   # Shifted +160.59
    'cw_y': 0.0,
    'x_whl': 618.42,  # Shifted +160.59
    'w_y': 480.0,
    'm_stat': 145.08
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# 2. Reset functionality (clears UI widget memory too)
def reset_sliders():
    for key, val in defaults.items():
        st.session_state[key] = val
        st.session_state[f"{key}_sl"] = val
        st.session_state[f"{key}_num"] = val

st.sidebar.button("🔄 Reset to Original Values", on_click=reset_sliders, type="primary")
st.sidebar.markdown("---")

# 3. Helper function to create a synced Slider + Text Box
def custom_slider(label, min_v, max_v, key, step_val):
    st.sidebar.markdown(f"**{label}**")
    
    sl_key = f"{key}_sl"
    num_key = f"{key}_num"
    
    # First-time setup for the UI keys
    if sl_key not in st.session_state:
        st.session_state[sl_key] = float(st.session_state[key])
        st.session_state[num_key] = float(st.session_state[key])

    # Callbacks to keep them perfectly synced when you interact with either
    def sync_from_slider():
        st.session_state[num_key] = st.session_state[sl_key]
        st.session_state[key] = st.session_state[sl_key]
        
    def sync_from_num():
        st.session_state[sl_key] = st.session_state[num_key]
        st.session_state[key] = st.session_state[num_key]

    # Layout: Slider takes 70% of the width, Input box takes 30%
    col1, col2 = st.sidebar.columns([7, 3])
    with col1:
        st.slider(label, float(min_v), float(max_v), key=sl_key, step=float(step_val), on_change=sync_from_slider, label_visibility="collapsed")
    with col2:
        st.number_input(label, float(min_v), float(max_v), key=num_key, step=float(step_val), on_change=sync_from_num, label_visibility="collapsed")
        
    return st.session_state[key]

# 4. Generate the linked UI elements
cw_m = custom_slider('CW Mass (kg) [Orig: 63.4]', 0.0, 100.0, 'cw_m', 0.5)
cw_z = custom_slider('CW Height Z (mm) [Orig: 178.0]', 10.0, 500.0, 'cw_z', 5.0)
cw_x = custom_slider('CW Pos X (mm) [Orig: 319.6]', 0.0, 800.0, 'cw_x', 5.0)
cw_y = custom_slider('CW Pos Y (mm) [Orig: 0.0]', -200.0, 200.0, 'cw_y', 5.0)

st.sidebar.markdown("---")

x_whl = custom_slider('Front Wheel X [Orig: 618.4]', 400.0, 1200.0, 'x_whl', 10.0)
w_y = custom_slider('Wheelbase Y (mm) [Orig: 480.0]', 300.0, 900.0, 'w_y', 10.0)

st.sidebar.markdown("---")

m_stat = custom_slider('Static Mass (kg) [Orig: 145.1]', 100.0, 200.0, 'm_stat', 1.0)


# --- MATH ENGINE ---
def calculate_fs(cw_m, cw_x, cw_z, cw_y, w_y, x_whl, m_stat):
    m_new = m_stat + cw_m
    weight_N = m_new * 9.80665
    
    com_x_new = (m_stat * com_x_stat + cw_m * cw_x) / m_new
    com_y_new = (m_stat * com_y_stat + cw_m * cw_y) / m_new
    com_z_new = (m_stat * com_z_stat + cw_m * cw_z) / m_new

    # 1. Normal Overbalance
    x_dist_1 = (w_y / 2) - com_y_new
    fs1 = ((np.arctan(x_dist_1 / com_z_new) * 180 / np.pi)) / 10

    # 2. Horizontal Instability
    fs2 = ((weight_N * x_dist_1) / 1291.94) / 150

    # 3. Vertical Instability (Updated for Max Reach COM Shift & Dynamic hd)
    com_x_extended = (m_stat * com_x_stat + cw_m * cw_x + 7062.4) / m_new
    x1_3 = x_whl - com_x_extended
    hd_dynamic = 1202.60 - x_whl  # Arm tip fixed position updated for +160.59 shift
    
    if x1_3 > 0 and hd_dynamic > 0:
        fs3 = ((weight_N * x1_3) / hd_dynamic) / 800
    else:
        fs3 = 0

    # 4. 10mm Threshold Test 
    hc_prime = com_z_new - 50.8
    r_prime_m = np.sqrt((x_whl - com_x_new)**2 + hc_prime**2) / 1000
    beta_rad = np.arctan((x_whl - com_x_new) / hc_prime)
    delta_pe_prime = weight_N * r_prime_m * (1 - np.cos(beta_rad))
    ke = 0.5 * m_new * (0.8**2)
    fs4 = delta_pe_prime / ke

    # 5. Impact Test
    b_dist = (w_y / 2) - com_y_new
    r_impact_m = np.sqrt(b_dist**2 + com_z_new**2) / 1000
    phi_impact_rad = np.arctan(b_dist / com_z_new)
    delta_pe_impact = weight_N * r_impact_m * (1 - np.cos(phi_impact_rad))
    inertia = m_new * (r_impact_m**2)
    j_impulse = np.sqrt(2 * inertia * delta_pe_impact) / 0.97406
    fs5 = j_impulse / 75

    return [fs1, fs2, fs3, fs4, fs5], m_new, [com_x_new, com_y_new, com_z_new]

def get_min_weight(cw_x, cw_z, cw_y, w_y, x_whl, m_stat):
    # Reduced step size to 0.1 for completely smooth plots
    for test_m in np.arange(0, 100, 0.1): 
        fs_vals, m_test, _ = calculate_fs(test_m, cw_x, cw_z, cw_y, w_y, x_whl, m_stat)
        if all(fs >= MIN_FS for fs in fs_vals):
            return m_test * 2.20462  
    return np.nan

# Calculate current state
current_fs, m_new, com_new = calculate_fs(cw_m, cw_x, cw_z, cw_y, w_y, x_whl, m_stat)

# --- METRICS DISPLAY ---
col1, col2, col3, col4, col5 = st.columns(5)
status = lambda fs: "✅ PASS" if fs >= MIN_FS else "❌ FAIL"
orig = [2.0, 2.5, 1.2, 2.2, 1.8]
col1.metric("T1 (Incline)", f"{current_fs[0]:.2f}", f"Orig: {orig[0]}", delta_color="off")
col2.metric("T2 (Push Top)", f"{current_fs[1]:.2f}", f"Orig: {orig[1]}", delta_color="off")
col3.metric("T3 (Push Arm)", f"{current_fs[2]:.2f}", f"Orig: {orig[2]}", delta_color="off")
col4.metric("T4 (Obstacle)", f"{current_fs[3]:.2f}", f"Orig: {orig[3]}", delta_color="off")
col5.metric("T5 (Shove)", f"{current_fs[4]:.2f}", f"Orig: {orig[4]}", delta_color="off")

st.markdown(f"**Total Mass:** {m_new:.1f} kg ({(m_new*2.20462):.1f} lbs) | **COM (x,y,z):** [{com_new[0]:.1f}, {com_new[1]:.1f}, {com_new[2]:.1f}]")

# --- PLOTTING ---
fig = plt.figure(figsize=(18, 10))
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.25, hspace=0.35)

ax_side = plt.subplot(2, 3, 1)
ax_front = plt.subplot(2, 3, 2)
ax_bar = plt.subplot(2, 3, 3)
ax_curve1 = plt.subplot(2, 2, 3)
ax_curve2 = plt.subplot(2, 2, 4)

# Widened X-limits to fit the newly shifted coordinate system
ax_side.set_title("Side Profile (X-Z)"); ax_side.set_xlim(-100, 1300); ax_side.set_ylim(-50, 1450); ax_side.grid(True, linestyle='--')
ax_front.set_title("Front Profile (Y-Z)"); ax_front.set_xlim(-600, 600); ax_front.set_ylim(-50, 1450); ax_front.grid(True, linestyle='--')
ax_bar.set_title("Current Safety Factors"); ax_bar.grid(axis='y', linestyle='--')
ax_curve1.set_title("Wheelbase Expansion vs. Min Weight"); ax_curve1.set_xlabel('Added "d" to Wheelbase (mm)'); ax_curve1.set_ylabel("Min Weight (lbs)"); ax_curve1.grid(True, linestyle='--')
ax_curve2.set_title("CW Repositioning vs. Min Weight"); ax_curve2.set_xlabel('CW Shift "d" (mm)'); ax_curve2.set_ylabel("Min Weight (lbs)"); ax_curve2.grid(True, linestyle='--')

# --- CHASSIS DRAWING ---
rear_wheel_x = 0.0
shift = 160.59 # Matches the coordinate shift for visual alignment

# Side Profile
ax_side.add_patch(patches.Rectangle((rear_wheel_x, 20), x_whl - rear_wheel_x, 130, linewidth=2, edgecolor='black', facecolor='darkgray', alpha=0.7))
ax_side.add_patch(patches.Rectangle((20 + shift, 150), 560, 800, linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.5))
ax_side.add_patch(patches.Rectangle((20 + shift, 850), 700, 150, linewidth=2, edgecolor='black', facecolor='whitesmoke', alpha=0.9))

# Front Profile
ax_front.add_patch(patches.Rectangle((-w_y/2, 20), w_y, 100, linewidth=2, edgecolor='black', facecolor='darkgray', alpha=0.7))
ax_front.add_patch(patches.Rectangle((-160, 120), 320, 830, linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.8))
ax_front.add_patch(patches.Rectangle((-120, 200), 240, 500, linewidth=1, edgecolor='gray', facecolor='dimgray', alpha=0.6))
ax_front.add_patch(patches.Rectangle((-200, 950), 400, 300, linewidth=2, edgecolor='darkcyan', facecolor='cadetblue', alpha=0.9))

# Counterweight
cw_w, cw_h = 120, 80
ax_side.add_patch(patches.Rectangle((cw_x - cw_w/2, cw_z - cw_h/2), cw_w, cw_h, linewidth=2, edgecolor='navy', facecolor='blue', alpha=0.9))
ax_front.add_patch(patches.Rectangle((cw_y - cw_w/2, cw_z - cw_h/2), cw_w, cw_h, linewidth=2, edgecolor='navy', facecolor='blue', alpha=0.9))

# Caster Wheels (2-inch / 50.8 mm radius)
wheel_r = 50.8
ax_side.add_patch(patches.Circle((rear_wheel_x, wheel_r), wheel_r, linewidth=2, edgecolor='darkred', facecolor='red', zorder=5))
ax_side.add_patch(patches.Circle((x_whl, wheel_r), wheel_r, linewidth=2, edgecolor='darkred', facecolor='red', zorder=5))

ax_front.add_patch(patches.Circle((-w_y/2, wheel_r), wheel_r, linewidth=2, edgecolor='darkred', facecolor='red', zorder=5))
ax_front.add_patch(patches.Circle((w_y/2, wheel_r), wheel_r, linewidth=2, edgecolor='darkred', facecolor='red', zorder=5))

# Center of Mass Markers
ax_side.plot(com_new[0], com_new[2], 'X', markersize=14, color='lime', markeredgecolor='black', zorder=10)
ax_front.plot(com_new[1], com_new[2], 'X', markersize=14, color='lime', markeredgecolor='black', zorder=10)

# Safety Factors Bar Chart
tests = ['Incline', 'Push Top', 'Push Arm', 'Obstacle', 'Shove']
bars = ax_bar.bar(tests, current_fs, color=['mediumseagreen' if fs >= MIN_FS else 'crimson' for fs in current_fs])
ax_bar.axhline(MIN_FS, color='black', linestyle='--', linewidth=2, label='Min (1.2)')
ax_bar.legend()

# Math Optimization Curves
d_vals_wb = np.arange(0, 201, 20)
d_vals_cw = np.arange(-400, 1, 20)
wt_wb = [get_min_weight(cw_x, cw_z, cw_y, w_y + d, x_whl + d, m_stat) for d in d_vals_wb]
wt_cwx = [get_min_weight(cw_x + d, cw_z, cw_y, w_y, x_whl, m_stat) for d in d_vals_cw]
wt_cwz = [get_min_weight(cw_x, max(cw_z + d, 10), cw_y, w_y, x_whl, m_stat) for d in d_vals_cw]

ax_curve1.plot(d_vals_wb, wt_wb, 'b-', linewidth=3)
ax_curve1.axhline(350, color='red', linestyle='--', label='Target (350 lbs)'); ax_curve1.legend()
ax_curve1.set_xlim(0, 200)

ax_curve2.plot(d_vals_cw, wt_cwx, 'g-', linewidth=3, label='X Shift (Backward)')
ax_curve2.plot(d_vals_cw, wt_cwz, 'purple', linewidth=3, label='Z Shift (Lower)')
ax_curve2.axhline(350, color='red', linestyle='--', label='Target (350 lbs)'); ax_curve2.legend()
ax_curve2.set_xlim(-400, 0)

st.pyplot(fig)
