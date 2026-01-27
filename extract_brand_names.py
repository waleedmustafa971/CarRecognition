import scipy.io
import json

# Load the mapping file
mat_data = scipy.io.loadmat('datasets/compcars/misc/make_model_name.mat')

# Extract make names
make_names = mat_data['make_names']

# Create mapping dictionary
brand_mapping = {}

print("Brand ID -> Brand Name mapping:\n")
for i, make in enumerate(make_names, start=1):
    brand_name = make[0][0]  # Extract the string from nested array
    brand_mapping[str(i)] = brand_name
    print(f"{i}: {brand_name}")

# Save to JSON
with open('brand_mapping.json', 'w') as f:
    json.dump(brand_mapping, f, indent=2)

print(f"\n✅ Saved {len(brand_mapping)} brands to brand_mapping.json")