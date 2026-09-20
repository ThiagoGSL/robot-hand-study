import xml.etree.ElementTree as ET

tree = ET.parse('dexhand-right.xml')
root = tree.getroot()
worldbody = root.find('worldbody')

# Create a new body for the hand, slightly above the floor
hand_body = ET.Element('body', name='hand_root', pos='0 0 0.001', euler='0 0 0')
freejoint = ET.Element('freejoint')
hand_body.append(freejoint)

# Move all children of worldbody to hand_body
for child in list(worldbody):
    hand_body.append(child)
    worldbody.remove(child)

# Remove all <inertial> tags so MuJoCo computes stable, realistic inertias from the geoms
for body in hand_body.iter('body'):
    for inertial in body.findall('inertial'):
        body.remove(inertial)

worldbody.append(hand_body)

# Add actuators for all joints found in the model
actuator = ET.Element('actuator')
# Find all joints inside the hand_body tree (excluding the freejoint we just added)
for joint in hand_body.iter('joint'):
    jname = joint.get('name')
    if jname:
        # The fingers are very light, so kp=0.06 is stable for them.
        # But the wrist has to lift the entire hand! It needs a much stronger kp.
        kp_val = '0.5' if 'wrist' in jname else '0.06'
        kv_val = '0.05' if 'wrist' in jname else '0.006'
        
        # Create a position actuator for this joint with real-world torque limits (1.5 N.m)
        pos_act = ET.Element('position', name=f"act_{jname}", joint=jname, kp=kp_val, kv=kv_val, forcelimited='true', forcerange='-1.5 1.5')
        
        # Apply the physical joint range to the actuator control range
        jrange = joint.get('range')
        if jrange:
            pos_act.set('ctrllimited', 'true')
            pos_act.set('ctrlrange', jrange)
            
        actuator.append(pos_act)

if len(actuator) > 0:
    root.append(actuator)

tree.write('dexhand-right.xml')
print("Added freejoint and actuators successfully.")
