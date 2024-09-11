import numpy as np

f = np.array([2, 3, 8, 15, 20, 16, 9, 7, 3, 1])
Hi = np.cumsum(f)

img = np.zeros(len(f))

MN = 84.0

q_low, q_high = 0.1, 0.1

# a_prim_low  = np.floor(MN*q_low)

# a_prim_high = np.floor(MN*(1-q_high))

a_prim_low = 0
a_prim_high = 255


for i, value in enumerate(Hi):
    if value >= np.round(MN*q_low):
        a_prim_low = i
        break

for i, value in enumerate(Hi):
    if value >= np.round(MN*(1-q_high)):
        a_prim_high = i
        break

print(np.round(MN*q_low))
print(np.round(MN*(1-q_high)))

print(a_prim_low)
print(a_prim_high)

a_prim_high = 6

a_min = 0
a_max = 255

for i, value in enumerate(f):
    if i <= a_prim_low:
        img[i] = a_min
    elif a_prim_low < i < a_prim_high:
        img[i] = np.round(a_min+(i-a_prim_low)*((a_max-a_min)/(a_prim_high-a_prim_low)))
    elif i >= a_prim_high:
        img[i] = a_max

print(img)
print(Hi)


