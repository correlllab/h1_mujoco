import finger_force_viz as fv
viz = fv.FingerForceVisualizer()
viz.setup_simulation()
for _ in range(500):
    viz.step()

state = viz.current_state
print(f'Contacts: {len(state.contacts)}, FC: {state.force_closure}, Q: {state.ferrari_canny:.6f}')
print()
print('Contact vs Sensor comparison (both are force ON finger):')
for finger, comp in state.comparison.items():
    cf = comp['contact_force_world']
    sf = comp['sensor_force_world']
    err = comp['force_error']
    nc = comp['num_contacts']
    print(f'  {finger:>8s} ({nc}c):  contact=[{cf[0]:7.4f},{cf[1]:7.4f},{cf[2]:7.4f}]  '
          f'sensor=[{sf[0]:7.4f},{sf[1]:7.4f},{sf[2]:7.4f}]  err={err:.4f}')
print()
print('Torque comparison:')
for finger, comp in state.comparison.items():
    ct = comp['contact_torque_world']
    st = comp['sensor_torque_world']
    err = comp['torque_error']
    print(f'  {finger:>8s}:  contact=[{ct[0]:8.5f},{ct[1]:8.5f},{ct[2]:8.5f}]  '
          f'sensor=[{st[0]:8.5f},{st[1]:8.5f},{st[2]:8.5f}]  err={err:.5f}')
