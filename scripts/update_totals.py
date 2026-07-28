import csv

csv_file = 'HGS_Partition_Costs.csv'
with open(csv_file, 'r') as f:
    reader = list(csv.reader(f))

header = reader[0]
rows = reader[1:]

instances = {}
for row in rows:
    if len(row) < 5: continue
    inst, chunk, cost, time, status = row
    if chunk == '[TOTAL AGGREGATE]': continue
    
    if inst not in instances:
        instances[inst] = {'total_chunks': 0, 'ok_chunks': 0, 'total_cost': 0.0, 'total_time': 0.0}
    
    instances[inst]['total_chunks'] += 1
    if status.startswith('OK'):
        instances[inst]['ok_chunks'] += 1
        try:
            instances[inst]['total_cost'] += float(cost)
        except ValueError:
            pass
        try:
            instances[inst]['total_time'] += float(time)
        except ValueError:
            pass

new_rows = []
for row in rows:
    if len(row) < 5: 
        new_rows.append(row)
        continue
    
    inst, chunk, cost, time, status = row
    if chunk == '[TOTAL AGGREGATE]':
        data = instances.get(inst, {'total_chunks': 0, 'ok_chunks': 0, 'total_cost': 0.0, 'total_time': 0.0})
        
        if data['total_chunks'] > 0 and data['ok_chunks'] == data['total_chunks']:
            agg_cost = f"{data['total_cost']:.2f}"
            agg_status = f"All {data['total_chunks']} succeeded"
        else:
            agg_cost = "INCOMPLETE"
            agg_status = f"Only {data['ok_chunks']}/{data['total_chunks']} succeeded"
            
        new_rows.append([inst, '[TOTAL AGGREGATE]', agg_cost, f"{data['total_time']:.2f}", agg_status])
    else:
        new_rows.append(row)

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(new_rows)

print("Updated aggregate rows in HGS_Partition_Costs.csv")
