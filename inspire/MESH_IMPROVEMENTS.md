# Inspire Hand Mesh Improvements Plan

## Overview

Two hand models exist for the Inspire hand, each with distinct trade-offs:

| Aspect | Option 1: `right_ur5_mount.xml` | Option 2: `inspire_right.xml` |
|--------|--------------------------------|------------------------------|
| Target | MuJoCo Warp (MJX) | Native MuJoCo |
| Collision Geometry | Simplified capsules/boxes | High-fidelity STL meshes |
| Contact Handling | Explicit pairs via `<contact>` | Automatic via contype/conaffinity |
| Controllability | Stable (kp=300, armature=1.0) | Unstable (kp=10, no armature) |
| Collision Coverage | Underestimated | Overestimated |

---

## Current Issues

### Option 2 Issues (inspire_right.xml)
1. **Jittery simulation**: Large contact points from overestimated collision meshes create excessive contact forces
2. **Thumb uncontrollable**: Thumb yaw/pitch joints exhibit instability due to:
   - Collision mesh interpenetration causing large impulses
   - Low kp=10 insufficient to counteract contact forces
   - No joint armature or damping
3. **Self-collision chaos**: No `<exclude>` pairs defined between adjacent links that naturally overlap
4. **Palm collision**: Large box primitives on base (lines 63-70) likely intersect with finger bases

### Option 1 Issues (right_ur5_mount.xml)
1. **Sticky contacts**: Objects appear to "stick" to hand during manipulation
2. **Unrealistic contact pairs**: Simplified capsules don't match actual finger geometry
3. **Scalability**: Manual contact pairs required per object

---

## Improvement Plan: Option 2 (inspire_right.xml)

### Goal
Make the high-fidelity model usable in native MuJoCo/MJPC by resolving self-collision and stability issues.

### Phase 1: Exclude Self-Collisions (Priority: HIGH)

Add `<contact>` section with `<exclude>` pairs for adjacent/overlapping links:

```xml
<contact>
  <!-- Palm to finger proximal exclusions -->
  <exclude body1="base" body2="thumb_proximal_base"/>
  <exclude body1="base" body2="index_proximal"/>
  <exclude body1="base" body2="middle_proximal"/>
  <exclude body1="base" body2="ring_proximal"/>
  <exclude body1="base" body2="pinky_proximal"/>

  <!-- Thumb chain exclusions -->
  <exclude body1="thumb_proximal_base" body2="thumb_proximal"/>
  <exclude body1="thumb_proximal" body2="thumb_intermediate"/>
  <exclude body1="thumb_intermediate" body2="thumb_distal"/>

  <!-- Finger chain exclusions (each finger) -->
  <exclude body1="index_proximal" body2="index_intermediate"/>
  <exclude body1="middle_proximal" body2="middle_intermediate"/>
  <exclude body1="ring_proximal" body2="ring_intermediate"/>
  <exclude body1="pinky_proximal" body2="pinky_intermediate"/>

  <!-- Cross-finger exclusions at palm level -->
  <exclude body1="index_proximal" body2="middle_proximal"/>
  <exclude body1="middle_proximal" body2="ring_proximal"/>
  <exclude body1="ring_proximal" body2="pinky_proximal"/>

  <!-- Thumb-to-palm exclusions -->
  <exclude body1="thumb_proximal_base" body2="index_proximal"/>
</contact>
```

### Phase 2: Reduce Collision Mesh Size (Priority: HIGH)

The current collision meshes in `assets/collision/` have "overestimated coverage". Options:

**Option A: Shrink existing meshes (Recommended first try)**
- Use a mesh processing tool (Blender, MeshLab) to:
  - Apply uniform scale of 0.90-0.95 to collision meshes
  - OR erode/shrink mesh by 0.5-1mm
- This preserves mesh shape while reducing interpenetration

**Option B: Replace with primitives**
- Replace mesh collision geoms with capsules/boxes like Option 1
- More work but guarantees no interpenetration
- Can be derived from the current primitive placements in Option 1

**Option C: Convex decomposition**
- Use VHACD or CoACD to create tighter convex hulls
- Better coverage than pure primitives, less problematic than full meshes

### Phase 3: Improve Joint Dynamics (Priority: MEDIUM)

Enable the commented-out dynamics in lines 13-14:

```xml
<default class="inspire">
  <position kp="100" dampratio="1.0" inheritrange="1" />
  <joint damping="0.1" armature="0.01" frictionloss="0.001" />
  <!-- ... -->
</default>
```

**Tuning guidelines:**
- `kp`: Start with 50-100, increase if fingers don't track well
- `dampratio`: 0.5-1.0 for smooth response without oscillation
- `armature`: 0.01-0.1 adds rotational inertia, stabilizes small joints
- `damping`: 0.05-0.2 adds viscous resistance
- `frictionloss`: Small value (0.001) for static friction

### Phase 4: Contact Parameters (Priority: MEDIUM)

Add contact-specific parameters to reduce force spikes:

```xml
<default class="collision">
  <geom
    type="mesh"
    group="3"
    condim="4"
    friction="1 0.005 0.001"
    solref="0.002 1"
    solimp="0.9 0.95 0.001"
  />
</default>
```

**Parameters explained:**
- `condim="4"`: Full friction cone (tangent + rolling)
- `solref="0.002 1"`: Softer contact (larger timeconst), damping ratio 1
- `solimp`: Contact impedance - smoother force ramping
- `friction`: Lower rolling/torsional friction

---

## Improvement Plan: Option 1 (right_ur5_mount.xml)

### Goal
Improve contact realism while maintaining MJX compatibility and explicit contact pairs.

### Phase 1: Diagnose Sticky Contacts (Priority: HIGH)

**Likely causes:**
1. `condim="1"` in default (line 12) - frictionless contacts lack tangent forces
2. Very stiff `solref=".001 1"` in task_inspire.xml (line 28)
3. Missing rolling/torsional friction

**Fix contact parameters in the hand model:**
```xml
<default class="inspire">
  <geom
    density="800"
    condim="4"
    contype="0"
    conaffinity="0"
    friction="0.8 0.005 0.001"
    solref="0.002 1"
  />
</default>
```

**Fix in task_inspire.xml:**
```xml
<default>
  <geom solref="0.002 1" solimp="0.9 0.95 0.001"/>
</default>
```

### Phase 2: Improve Collision Primitive Coverage (Priority: MEDIUM)

Current capsules are thin (size 0.005) and sparse. Improve coverage:

1. **Add more capsules per finger segment**
   - Current: 1 capsule per segment
   - Target: 2-3 capsules to cover finger width

2. **Adjust capsule radii**
   - Increase from 0.005 to 0.007-0.008 for fingertips
   - Maintain 0.01 for thumb

3. **Add palm collision grid**
   - Current: Single box at `0 -0.1 0`
   - Add: Multiple smaller boxes or ellipsoids covering palm surface

**Example improved finger collision:**
```xml
<geom name="collision_hand_right_index_1a" type="capsule"
  fromto="-0.002 -0.009 -0.003 0.002 0.032 -0.003"
  size="0.006" contype="0" conaffinity="0" group="3" density="0"/>
<geom name="collision_hand_right_index_1b" type="capsule"
  fromto="-0.002 -0.009 -0.007 0.002 0.032 -0.007"
  size="0.006" contype="0" conaffinity="0" group="3" density="0"/>
```

### Phase 3: Auto-Generate Contact Pairs (Priority: LOW)

Create a Python script to generate contact pairs:

```python
def generate_contact_pairs(hand_geoms: list[str], object_geom: str) -> str:
    """Generate <contact> XML for hand-object pairs."""
    pairs = []
    for geom in hand_geoms:
        pairs.append(f'<pair geom1="{object_geom}" geom2="{geom}"/>')
    return '\n'.join(pairs)

HAND_COLLISION_GEOMS = [
    "collision_hand_right_palm_0",
    "collision_hand_right_thumb_0", "collision_hand_right_thumb_1",
    "collision_hand_right_index_0", "collision_hand_right_index_1",
    "collision_hand_right_middle_0", "collision_hand_right_middle_1",
    "collision_hand_right_ring_0", "collision_hand_right_ring_1",
    "collision_hand_right_pinky_0", "collision_hand_right_pinky_1",
]
```

---

## Validation Checklist

### For Option 2:
- [ ] Hand at rest (no control) is stable, no jitter
- [ ] Thumb yaw/pitch responds to control inputs
- [ ] Fingers close without self-penetration
- [ ] Contact with objects produces expected forces (check contact force sensor)
- [ ] No link separation or explosion on contact

### For Option 1:
- [ ] Objects don't stick after release (gravity pulls them away)
- [ ] Pinch grasp holds object stably
- [ ] Contact forces are smooth, not spiky
- [ ] Objects slide appropriately with friction

---

## Recommended Testing Sequence

1. **Static stability test**: Load model, set home pose, run 1000 steps with no control
2. **Control response test**: Apply sinusoidal control to each joint, verify tracking
3. **Self-collision test**: Close all fingers fully, check for explosions
4. **Object manipulation test**: Grasp and lift object, check for sticking/slipping
5. **Force profile test**: Log contact forces during manipulation, check for spikes

---

## Resources

- MuJoCo XML Reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html
- MuJoCo Contact Model: https://mujoco.readthedocs.io/en/stable/modeling.html#contact
- VHACD for convex decomposition: https://github.com/kmammou/v-hacd
- CoACD (better for thin structures): https://github.com/SarahWeiii/CoACD
