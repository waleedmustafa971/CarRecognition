import json
import yaml

# Load the brand mapping
with open('brand_mapping.json', 'r') as f:
    brand_mapping = json.load(f)

# Create the corrected names list (in order from 1 to 163)
correct_names = [brand_mapping[str(i)] for i in range(1, 164)]

# Load current data.yaml
with open('datasets/compcars_yolo/data.yaml', 'r') as f:
    data = yaml.safe_load(f)

# Update the names
data['names'] = correct_names

# Save corrected data.yaml
with open('datasets/compcars_yolo/data.yaml', 'w') as f:
    yaml.dump(data, f, sort_keys=False)

print("✅ Updated data.yaml with correct brand names!")
print(f"\nFirst 10 brands:")
for i in range(10):
    print(f"  {i}: {correct_names[i]}")