import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Define colors (all text will be black)
black = "black"

# --- Parameters ---
natural_freq = 10 # Example natural frequency in GHz (visual scale)
# We'll use a conceptual detuning for visual scaling, not the physical one
# Let's say we want Delta to look like 1/4 of omega_0's height
conceptual_detuning_ratio = 0.25 # 25% of omega_0's visual height

# --- Function to draw a single energy level diagram ---
def draw_energy_level_diagram(ax, title, is_target=True):
    # Energy levels (adjust Y-positions for visual separation)
    level_0_y = 0.1
    level_1_y = 0.9
    level_x_center = 0.5
    level_width = 0.6
    
    total_height = level_1_y - level_0_y

    # Draw levels
    ax.hlines(level_0_y, level_x_center - level_width/2, level_x_center + level_width/2, color=black, linewidth=2)
    ax.hlines(level_1_y, level_x_center - level_width/2, level_x_center + level_width/2, color=black, linewidth=2)

    # Level labels (black text)
    ax.text(level_x_center - level_width/2 - 0.1, level_0_y, r'$|0\rangle$', va='center', ha='right', fontsize=14, color=black)
    ax.text(level_x_center - level_width/2 - 0.1, level_1_y, r'$|1\rangle$', va='center', ha='right', fontsize=14, color=black)

    # Natural frequency (omega_0) arrow (black text and arrow)
    ax.annotate("", xy=(level_x_center + level_width/2 + 0.05, level_1_y),
                xytext=(level_x_center + level_width/2 + 0.05, level_0_y),
                arrowprops=dict(facecolor=black, edgecolor=black, arrowstyle='<->', linewidth=1.5),
                annotation_clip=False)
    ax.text(level_x_center + level_width/2 + 0.1, (level_0_y + level_1_y)/2, r'$\omega_0$', va='center', ha='left', fontsize=16, color=black)

    # Detuning details for spectator only
    if not is_target:
        # Calculate the visual 'height' for the dashed line based on the conceptual_detuning_ratio
        conceptual_driven_y = level_1_y - (total_height * conceptual_detuning_ratio)

        # Dashed line for the effective driven frequency (representing where omega_d would resonant)
        ax.hlines(conceptual_driven_y, level_x_center - level_width/2, level_x_center + level_width/2,
                  color='grey', linestyle='--', linewidth=1)

        # Arrow for Delta
        arrow_x_pos = level_x_center + level_width/2 + 0.2
        ax.annotate("", xy=(arrow_x_pos, level_1_y),
                    xytext=(arrow_x_pos, conceptual_driven_y + 0.01), # Small offset for clarity
                    arrowprops=dict(facecolor=black, edgecolor=black, arrowstyle='<->', linewidth=1),
                    annotation_clip=False)
        ax.text(arrow_x_pos + 0.05, (level_1_y + conceptual_driven_y)/2, r'$\Delta$', va='center', ha='left', fontsize=16, color=black)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off() # Hide axes
    ax.set_title(title, color=black, fontsize=18, pad=20) # Title text is black

# --- Plot 1: Target Qubit ---
fig1, ax1 = plt.subplots(figsize=(4, 5))
draw_energy_level_diagram(ax1, "Target Qubit", is_target=True)
plt.show()

# --- Plot 2: Spectator Qubit ---
fig2, ax2 = plt.subplots(figsize=(4, 5))
draw_energy_level_diagram(ax2, "Spectator Qubit", is_target=False)
plt.show()