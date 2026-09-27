from ase.db import connect

db = connect("train.db")

row = db.get(id=1)

latt_dis = eval(row.latt_dis)
intensity = eval(row.intensity)
target = eval(row.tager)
simulation_param = eval(row.simulation_param)


print("====================================")
print(" First Training Entry")
print("====================================")

print("Formula :", row.chem_form)

print("\nTarget")
print("Space group    :", target[0])
print("Crystal system :", target[1])

print("\nXRD")
print("Number of d points :", len(latt_dis))
print("Number of I points :", len(intensity))

print("d start :", latt_dis[0])
print("d end   :", latt_dis[-1])

print("\nSimulation parameters")
print("Grain size            :", simulation_param[0])
print("Preferred orientation :", simulation_param[1])
print("Thermal vibration     :", simulation_param[2])
print("Zero shift            :", simulation_param[3])
