import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches

# --- 1. SYSTEM CONSTANTS & INITIAL PARAMETERS ---
m_stat_init = 145.08
com_x_stat = 159.0
com_z_stat = 858.0
com_y_stat = 1.81

cw_mass_init = 63.38  # Starting with original weight
cw_z_init = 178.0
cw_x_init = 159.0
cw_y_init = 0.0

w_y_init = 480.0
x_wheel_init = 457.83

MIN_FS = 1.2

# --- 2. MATH ENGINE ---
def calculate_fs(cw_m, cw_x, cw_z, cw_y, w_y, x_whl, m_stat):
    """Returns the 5 Safety Factors for a given configuration"""
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

    # 3. Vertical Instability (Updated for Max Reach COM & Dynamic 'hd')
    # The physical extension of the arm creates a constant moment shift in X.
    # Original Total Mass (208.45 kg) * Shift (192.78 - 158.90 mm) = 7062.4 kg*mm
    com_x_extended = (m_stat * com_x_stat + cw_m * cw_x + 7062.4) / m_new
    
    x1_3 = x_whl - com_x_extended
    hd_dynamic = 1042.01 - x_whl  # Distance from front wheels to fixed arm tip
    
    if x1_3 > 0 and hd_dynamic > 0:
        fs3 = ((weight_N * x1_3) / hd_dynamic) / 800
    else:
        fs3 = 0

    # 4. 10mm Threshold Test 
    # (Uses the normal parked COM position, NOT extended)
    x1_4 = x_whl - com_x_new
    hc_prime = com_z_new - 50.8
    r_prime_m = np.sqrt(x1_4**2 + hc_prime**2) / 1000
    beta_rad = np.arctan(x1_4 / hc_prime)
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
    """Sweeps CW mass to find the lowest possible system weight where all FS >= 1.2"""
    for cw_m in np.arange(0, 100, 1.0): 
        fs_vals, m_new, _ = calculate_fs(cw_m, cw_x, cw_z, cw_y, w_y, x_whl, m_stat)
        if all(fs >= MIN_FS for fs in fs_vals):
            return m_new * 2.20462  
    return np.nan

# --- 3. SETUP FIGURE & PLOTS ---
fig = plt.figure(figsize=(18, 10))
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.35, wspace=0.25, hspace=0.35)

# Top Row: Diagrams and Bar Chart
ax_side = plt.subplot(2, 3, 1)
ax_front = plt.subplot(2, 3, 2)
ax_bar = plt.subplot(2, 3, 3)

# Bottom Row: Trade-off Curves and Info Text
ax_curve1 = plt.subplot(2, 3, 4)
ax_curve2 = plt.subplot(2, 3, 5)
ax_info = plt.subplot(2, 3, 6)
ax_info.axis('off')

# --- CONFIGURE SUBPLOTS ---
ax_side.set_title("Side Profile (X-Z)")
ax_side.set_xlim(-200, 1000); ax_side.set_ylim(-50, 1450); ax_side.grid(True, linestyle='--', alpha=0.6)

ax_front.set_title("Front Profile (Y-Z)")
ax_front.set_xlim(-600, 600); ax_front.set_ylim(-50, 1450); ax_front.grid(True, linestyle='--', alpha=0.6)

ax_bar.set_title("Current Safety Factors")
ax_bar.set_ylabel("Factor of Safety")
ax_bar.grid(axis='y', linestyle='--', alpha=0.7)

ax_curve1.set_title("Wheelbase Expansion vs. Min Weight")
ax_curve1.set_xlabel('Added "d" to Current Wheelbase (mm)')
ax_curve1.set_ylabel("Min System Weight (lbs)")
ax_curve1.grid(True, linestyle='--', alpha=0.7)

ax_curve2.set_title("CW Repositioning vs. Min Weight")
ax_curve2.set_xlabel('CW Shift "d" from Current (mm)')
ax_curve2.set_ylabel("Min System Weight (lbs)")
ax_curve2.grid(True, linestyle='--', alpha=0.7)

# --- DRAW DYNAMIC SHAPES ---
chassis_base_side = patches.Rectangle((0, 20), x_wheel_init, 130, linewidth=2, edgecolor='black', facecolor='darkgray', alpha=0.7)
chassis_tower_side = patches.Rectangle((20, 150), 280, 800, linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.5)
chassis_arm_side = patches.Rectangle((20, 850), 700, 150, linewidth=2, edgecolor='black', facecolor='whitesmoke', alpha=0.9)
ax_side.add_patch(chassis_base_side); ax_side.add_patch(chassis_tower_side); ax_side.add_patch(chassis_arm_side)

chassis_base_front = patches.Rectangle((-w_y_init/2, 20), w_y_init, 100, linewidth=2, edgecolor='black', facecolor='darkgray', alpha=0.7)
chassis_tower_front = patches.Rectangle((-160, 120), 320, 830, linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.8)
chassis_window_front = patches.Rectangle((-120, 200), 240, 500, linewidth=1, edgecolor='gray', facecolor='dimgray', alpha=0.6)
chassis_top_front = patches.Rectangle((-200, 950), 400, 300, linewidth=2, edgecolor='darkcyan', facecolor='cadetblue', alpha=0.9)
ax_front.add_patch(chassis_base_front); ax_front.add_patch(chassis_tower_front); ax_front.add_patch(chassis_window_front); ax_front.add_patch(chassis_top_front)

cw_width, cw_height = 120, 80
cw_patch_side = patches.Rectangle((cw_x_init - cw_width/2, cw_z_init - cw_height/2), cw_width, cw_height, linewidth=2, edgecolor='navy', facecolor='blue', alpha=0.9)
cw_patch_front = patches.Rectangle((cw_y_init - cw_width/2, cw_z_init - cw_height/2), cw_width, cw_height, linewidth=2, edgecolor='navy', facecolor='blue', alpha=0.9)
ax_side.add_patch(cw_patch_side); ax_front.add_patch(cw_patch_front)

front_wheel_side = ax_side.plot(x_wheel_init, 0, '^', markersize=18, color='red')[0]
ax_side.plot(0, 0, '^', markersize=18, color='red') 
left_wheel_front = ax_front.plot(-w_y_init/2, 0, '^', markersize=18, color='red')[0]
right_wheel_front = ax_front.plot(w_y_init/2, 0, '^', markersize=18, color='red')[0]

com_point_side = ax_side.plot([], [], 'X', markersize=14, color='lime', markeredgecolor='black')[0]
com_point_front = ax_front.plot([], [], 'X', markersize=14, color='lime', markeredgecolor='black')[0]

# --- INITIALIZE PLOT OBJECTS ---
tests = ['Incline', 'Push Top', 'Push Arm', 'Obstacle', 'Shove']
bar_chart = ax_bar.bar(tests, [0,0,0,0,0], color='gray')
ax_bar.axhline(MIN_FS, color='black', linestyle='--', linewidth=2, label='Min (1.2)')
ax_bar.legend()

d_vals_wb = np.arange(0, 201, 20)      
d_vals_cw = np.arange(-400, 1, 20)     

line_wb, = ax_curve1.plot([], [], 'b-', linewidth=3)
ax_curve1.axhline(350, color='red', linestyle='--', label='Target (350 lbs)')
ax_curve1.legend()

line_cwx, = ax_curve2.plot([], [], 'g-', linewidth=3, label='X Shift (Backward)')
line_cwz, = ax_curve2.plot([], [], 'purple', linewidth=3, label='Z Shift (Lower)')
ax_curve2.axhline(350, color='red', linestyle='--', label='Target (350 lbs)')
ax_curve2.legend()

info_text = ax_info.text(0.1, 0.9, '', fontsize=12, family='monospace', va='top',
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

# --- SLIDERS SETUP (Spaced out to fit Red Line text) ---
axcolor = 'lightgoldenrodyellow'
slider_axes = [
    plt.axes([0.15, 0.23, 0.25, 0.02], facecolor=axcolor), 
    plt.axes([0.15, 0.18, 0.25, 0.02], facecolor=axcolor), 
    plt.axes([0.15, 0.13, 0.25, 0.02], facecolor=axcolor), 
    plt.axes([0.15, 0.08, 0.25, 0.02], facecolor=axcolor), 
    plt.axes([0.65, 0.23, 0.25, 0.02], facecolor=axcolor), 
    plt.axes([0.65, 0.18, 0.25, 0.02], facecolor=axcolor),
    plt.axes([0.65, 0.13, 0.25, 0.02], facecolor=axcolor)  
]

s_cw_mass = Slider(slider_axes[0], 'CW Mass (kg)', 0, 100, valinit=cw_mass_init)
s_cw_z = Slider(slider_axes[1], 'CW Height Z (mm)', 10, 500, valinit=cw_z_init)
s_cw_x = Slider(slider_axes[2], 'CW Pos X (mm)', 0, 400, valinit=cw_x_init)
s_cw_y = Slider(slider_axes[3], 'CW Pos Y (mm)', -200, 200, valinit=cw_y_init)
s_x_wheel = Slider(slider_axes[4], 'Front Wheel X Cooridnate (mm)', 300, 900, valinit=x_wheel_init)
s_w_y = Slider(slider_axes[5], 'Wheelbase Y (mm)', 300, 900, valinit=w_y_init)
s_m_stat = Slider(slider_axes[6], 'Static Mass (kg)', 100, 200, valinit=m_stat_init)

# Helper function to draw the red line and original text
def add_red_line(ax, orig_val, is_mass=False):
    ax.axvline(orig_val, color='red', linewidth=2, zorder=10) # Draws red line over the slider track
    if is_mass:
        label = f"{orig_val:.1f} ({orig_val*2.20462:.1f} lbs)"
    else:
        label = f"{orig_val:.1f}"
    ax.text(orig_val, 1.1, label, color='red', fontsize=9, fontweight='bold',
            transform=ax.get_xaxis_transform(), ha='center', va='bottom')

# Apply red lines to all sliders
add_red_line(slider_axes[0], cw_mass_init, is_mass=True)
add_red_line(slider_axes[1], cw_z_init)
add_red_line(slider_axes[2], cw_x_init)
add_red_line(slider_axes[3], cw_y_init)
add_red_line(slider_axes[4], x_wheel_init)
add_red_line(slider_axes[5], w_y_init)
add_red_line(slider_axes[6], m_stat_init, is_mass=True)


# --- UPDATE FUNCTION ---
def update(val):
    cw_m, cw_z, cw_x, cw_y = s_cw_mass.val, s_cw_z.val, s_cw_x.val, s_cw_y.val
    x_whl, w_y, m_stat = s_x_wheel.val, s_w_y.val, s_m_stat.val

    # Update live text displays for mass sliders to include lbs
    s_cw_mass.valtext.set_text(f"{cw_m:.1f} ({cw_m*2.20462:.1f} lbs)")
    s_m_stat.valtext.set_text(f"{m_stat:.1f} ({m_stat*2.20462:.1f} lbs)")

    # 1. Math Updates 
    current_fs, m_new, com_new = calculate_fs(cw_m, cw_x, cw_z, cw_y, w_y, x_whl, m_stat)
    
    # 2. Visual Diagram Updates
    chassis_base_side.set_width(x_whl)
    chassis_base_front.set_width(w_y)
    chassis_base_front.set_x(-w_y / 2)

    front_wheel_side.set_xdata([x_whl])
    left_wheel_front.set_xdata([-w_y / 2])
    right_wheel_front.set_xdata([w_y / 2])

    cw_patch_side.set_xy((cw_x - cw_width/2, cw_z - cw_height/2))
    cw_patch_front.set_xy((cw_y - cw_width/2, cw_z - cw_height/2))
    
    com_point_side.set_data([com_new[0]], [com_new[2]])
    com_point_front.set_data([com_new[1]], [com_new[2]])

    # 3. Bar Chart Updates
    for bar, fs in zip(bar_chart, current_fs):
        bar.set_height(fs)
        bar.set_color('mediumseagreen' if fs >= MIN_FS else 'crimson')
    ax_bar.set_ylim(0, max(max(current_fs) + 0.5, 3.5))

    # 4. Curve Generation 
    wt_wb, wt_cwx, wt_cwz = [], [], []
    for d in d_vals_wb:
        wt_wb.append(get_min_weight(cw_x, cw_z, cw_y, w_y + d, x_whl + d, m_stat))
    for d in d_vals_cw:
        wt_cwx.append(get_min_weight(cw_x + d, cw_z, cw_y, w_y, x_whl, m_stat))
        new_z = max(cw_z + d, 10) 
        wt_cwz.append(get_min_weight(cw_x, new_z, cw_y, w_y, x_whl, m_stat))

    line_wb.set_data(d_vals_wb, wt_wb)
    line_cwx.set_data(d_vals_cw, wt_cwx)
    line_cwz.set_data(d_vals_cw, wt_cwz)
    
    # Dynamically scale plot bounds
    ax_curve1.set_xlim(0, 200)
    all_wts_wb = [w for w in wt_wb if not np.isnan(w)]
    if all_wts_wb:
        ax_curve1.set_ylim(max(200, min(all_wts_wb) - 20), max(500, max(all_wts_wb) + 20))

    ax_curve2.set_xlim(-400, 0)
    all_wts_cw = [w for w in wt_cwx + wt_cwz if not np.isnan(w)]
    if all_wts_cw:
        ax_curve2.set_ylim(max(200, min(all_wts_cw) - 20), max(500, max(all_wts_cw) + 20))

    # 5. Text Box Updates 
    status = lambda fs, tgt: "PASS" if fs >= tgt else "FAIL"
    text_str = (
        f"==== CURRENT CONFIG ====\n"
        f"Total Mass: {m_new:.1f} kg ({(m_new*2.20462):.1f} lbs)\n"
        f"COM (x,y,z): [{com_new[0]:.1f}, {com_new[1]:.1f}, {com_new[2]:.1f}]\n\n"
        f"==== SAFETY FACTORS ====\n"
        f"T1 (Incline) : {current_fs[0]:.2f} [{status(current_fs[0], 2.0)}] (Orig: 2.0)\n"
        f"T2 (Push Top): {current_fs[1]:.2f} [{status(current_fs[1], 2.5)}] (Orig: 2.5)\n"
        f"T3 (Push Arm): {current_fs[2]:.2f} [{status(current_fs[2], 1.2)}] (Orig: 1.2)\n"
        f"T4 (Obstacle): {current_fs[3]:.2f} [{status(current_fs[3], 2.2)}] (Orig: 2.2)\n"
        f"T5 (Shove)   : {current_fs[4]:.2f} [{status(current_fs[4], 1.8)}] (Orig: 1.8)\n\n"
        f"NOTE: Curves below show potential\nweight reductions IF you modify\nthis specific configuration."
    )
    info_text.set_text(text_str)
    fig.canvas.draw_idle()

s_cw_mass.on_changed(update)
s_cw_z.on_changed(update)
s_cw_x.on_changed(update)
s_cw_y.on_changed(update)
s_x_wheel.on_changed(update)
s_w_y.on_changed(update)
s_m_stat.on_changed(update)

update(None)

plt.show()