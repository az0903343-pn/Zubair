"""
Generate a large synthetic vehicle dataset (cycle to helicopter).
Run: python generate_data.py
Output: data/vehicles_15000.csv  (use this for Kaggle training / model)
"""
import os
import random
import pandas as pd

random.seed(42)

BRANDS_BY_TYPE = {
    "Cycle": ["Hero", "Atlas", "BSA", "Raleigh", "Cannondale", "Trek", "Giant"],
    "Bike": ["Hero", "Honda", "Yamaha", "Bajaj", "TVS", "Royal Enfield", "KTM", "Suzuki"],
    "Scooter": ["Honda", "Yamaha", "Vespa", "Suzuki", "TVS", "Ather", "Ola"],
    "Car": ["Maruti", "Hyundai", "Tata", "Mahindra", "Honda", "Toyota", "Kia", "MG", "Ford"],
    "Electric": ["Tesla", "MG", "Hyundai", "Tata", "BYD", "Ather", "Ola", "Simple Energy"],
    "MPV": ["Toyota", "Mahindra", "Kia", "Maruti", "Hyundai"],
    "Off-Roader": ["Mahindra", "Force", "Jeep", "Toyota", "Land Rover"],
    "Van": ["Tata", "Maruti", "Hyundai", "Force", "Ashok Leyland"],
    "Rickshaw": ["Piaggio", "Bajaj", "TVS", "Mahindra", "Atul Auto"],
    "Helicopter": ["Robinson", "Airbus", "Bell", "Leonardo", "Sikorsky"],
}

TYPE_FUEL = {
    "Cycle": ["Human"],
    "Bike": ["Petrol", "Electric"],
    "Scooter": ["Petrol", "Electric"],
    "Car": ["Petrol", "Diesel", "Electric", "CNG"],
    "Electric": ["Electric"],
    "MPV": ["Diesel", "Petrol"],
    "Off-Roader": ["Diesel", "Petrol"],
    "Van": ["Diesel", "Petrol", "CNG"],
    "Rickshaw": ["Petrol", "CNG", "Electric"],
    "Helicopter": ["Aviation"],
}

PRICE_RANGE = {
    "Cycle": (60, 900),
    "Bike": (700, 4500),
    "Scooter": (800, 1600),
    "Car": (4000, 35000),
    "Electric": (8000, 60000),
    "MPV": (12000, 45000),
    "Off-Roader": (10000, 55000),
    "Van": (7000, 25000),
    "Rickshaw": (2000, 5500),
    "Helicopter": (300000, 2500000),
}

MILEAGE = {
    "Cycle": (0, 0),
    "Bike": (35, 90),
    "Scooter": (40, 60),
    "Car": (14, 26),
    "Electric": (120, 200),
    "MPV": (10, 16),
    "Off-Roader": (8, 14),
    "Van": (10, 16),
    "Rickshaw": (25, 45),
    "Helicopter": (0, 0),
}

SEATS = {
    "Cycle": (1, 1),
    "Bike": (1, 2),
    "Scooter": (2, 2),
    "Car": (4, 7),
    "Electric": (2, 5),
    "MPV": (6, 8),
    "Off-Roader": (4, 7),
    "Van": (6, 15),
    "Rickshaw": (3, 6),
    "Helicopter": (2, 8),
}

SAFETY = {
    "Cycle": (1, 3),
    "Bike": (2, 4),
    "Scooter": (2, 4),
    "Car": (3, 5),
    "Electric": (4, 5),
    "MPV": (3, 5),
    "Off-Roader": (3, 5),
    "Van": (3, 5),
    "Rickshaw": (1, 3),
    "Helicopter": (3, 5),
}

DESC_TEMPLATES = {
    "Cycle": "Durable {brand} cycle for daily commute and fitness.",
    "Bike": "Reliable {brand} bike with great mileage and city handling.",
    "Scooter": "Comfortable {brand} automatic scooter for urban rides.",
    "Car": "Spacious {brand} car with modern features and safety.",
    "Electric": "Zero-emission {brand} electric vehicle with low running cost.",
    "MPV": "Family-friendly {brand} MPV with flexible seating.",
    "Off-Roader": "Rugged {brand} off-roader built for tough terrain.",
    "Van": "Practical {brand} van for cargo or passenger transport.",
    "Rickshaw": "Commercial {brand} rickshaw for city transport business.",
    "Helicopter": "Private {brand} helicopter for executive travel.",
}

def random_price(lo, hi):
    return round(random.uniform(lo, hi), -2)

def main():
    n_per_type = 1500
    rows = []
    for vtype, brands in BRANDS_BY_TYPE.items():
        fuels = TYPE_FUEL[vtype]
        for i in range(n_per_type):
            brand = random.choice(brands)
            fuel = random.choice(fuels)
            price = random_price(*PRICE_RANGE[vtype])
            mileage = round(random.uniform(*MILEAGE[vtype]), 1)
            seats = random.randint(*SEATS[vtype])
            safety = random.randint(*SAFETY[vtype])
            # Advantage score: value for money + features
            adv = round(random.uniform(60, 95), 1)
            desc = DESC_TEMPLATES[vtype].format(brand=brand)
            rows.append({
                "Vehicle_ID": f"V{len(rows)+1:05d}",
                "Name": f"{brand} {vtype} Model {i+1}",
                "Brand": brand,
                "Type": vtype,
                "Price_USD": price,
                "Mileage_kmpl": mileage,
                "Seats": seats,
                "Fuel_Type": fuel,
                "Safety_Rating": safety,
                "Advantage_Score": adv,
                "Description": desc,
                "Link": f"https://example.com/{brand.lower()}-{vtype.lower()}-{i+1}",
            })

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    out = "data/vehicles_15000.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} vehicles -> {out}")
    print(df["Type"].value_counts())

if __name__ == "__main__":
    main()
