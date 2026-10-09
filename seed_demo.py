from pathlib import Path
from datetime import date, timedelta
import random
import pandas as pd

BASE=Path(__file__).resolve().parent
DATA=BASE/"data"; DATA.mkdir(exist_ok=True)
random.seed(42)

suppliers=["Alpha Components","Beta Industrial","Gamma Metals","Delta Logistics","Epsilon Tech"]
cats=["Electrical","Mechanical","Steel","Services"]
rows=[]
today=date.today()
for i in range(140):
    supplier=random.choices(suppliers,weights=[28,25,20,15,12])[0]
    cat=random.choice(cats)
    order=today-timedelta(days=random.randint(20,180))
    promised=order+timedelta(days=random.randint(10,45))
    delivered=random.random()<0.82
    # Alpha is intentionally worse; Beta stronger
    if delivered:
        late_p=0.48 if supplier=="Alpha Components" else 0.12 if supplier=="Beta Industrial" else 0.25
        delta=random.randint(1,18) if random.random()<late_p else -random.randint(0,4)
        actual=promised+timedelta(days=delta)
        status="CLOSED"
    else:
        actual=None
        status="OPEN"
        # ensure some overdue opens
        if random.random()<0.5:
            promised=today-timedelta(days=random.randint(1,35))
    qty=random.randint(10,500)
    price=round(random.uniform(8,950),2)
    received=qty if delivered else 0
    rejected=0
    if delivered and random.random() < (0.12 if supplier=="Gamma Metals" else 0.04):
        rejected=random.randint(1,max(1,int(qty*.12)))
    rows.append({
        "Purch.Doc.":f"45{100000+i:06d}","Item":10,"Vendor":f"V{suppliers.index(supplier)+1:03d}","Vendor Name":supplier,
        "Item_Category":cat,"Order_Date":order.isoformat(),"Deliv. Date":promised.isoformat(),"GR Date":actual.isoformat() if actual else None,
        "PO Qty":qty,"Received_Qty":received,"Defective_Units":rejected,"Net Price":price,"Curr.":"EUR","UOM":"EA","Order_Status":status,
        "Compliance":"Yes" if random.random()>.06 else "No"
    })

p=DATA/"demo_procurement.csv"
pd.DataFrame(rows).to_csv(p,index=False)
print(p)
