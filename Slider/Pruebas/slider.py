# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# Create function to be called when slider value is changed

def update(val):
    lower_h = l_h.val
    lower_s = l_s.val
    lower_v = l_v.val
    upper_h = u_h.val
    upper_s = u_s.val
    upper_v = u_v.val
 
# Create a function resetSlider to set slider to
# initial values when Reset button is clicked

def resetSlider(event):
    l_h.reset()
    l_s.reset()
    l_v.reset()
    u_h.reset()
    u_s.reset()
    u_v.reset()

# Create a subplot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
lower_h = 0.6
lower_s = 0.2
lower_v = 0.5
upper_h = 0.6
upper_s = 0.2
upper_v = 0.5



ax_lower_h = plt.axes([0.2, 0.2, 0.65, 0.03])
ax_lower_s = plt.axes([0.2, 0.17, 0.65, 0.03])
ax_lower_v = plt.axes([0.2, 0.14, 0.65, 0.03])
ax_upper_h = plt.axes([0.2, 0.11, 0.65, 0.03])
ax_upper_s = plt.axes([0.2, 0.08, 0.65, 0.03])
ax_upper_v = plt.axes([0.2, 0.05, 0.65, 0.03])


# Create a slider from 0.0 to 1.0 in axes axred
# with 0.6 as initial value.

l_h = Slider(ax_lower_h, 'Lower h', 0.0, 179.0, valinit=90)
l_s = Slider(ax_lower_s, 'Lower s', 0.0, 255.0, valinit=128)
l_v = Slider(ax_lower_v, 'Lower v', 0.0, 255.0, valinit=128)
u_h = Slider(ax_upper_h, 'Upper h', 0.0, 179.0, valinit=90)
u_s = Slider(ax_upper_s, 'Upper s', 0.0, 255.0, valinit=128)
u_v = Slider(ax_upper_v, 'Upper v', 0.0, 255.0, valinit=128)

# Call update function when slider value is changed
l_h.on_changed(update)
l_s.on_changed(update)
l_v.on_changed(update)
u_h.on_changed(update)
u_s.on_changed(update)
u_v.on_changed(update)

# Create axes for reset button and create button
resetax = plt.axes([0.8, 0.9, 0.1, 0.04])
button = Button(resetax, 'Reset', color='gold',
				hovercolor='skyblue')

# Call resetSlider function when clicked on reset button
button.on_clicked(resetSlider)

# Display graph
plt.show()


