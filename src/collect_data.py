import fastf1
import pandas as pd
import os

# Create cache folder if it doesn't exist
os.makedirs("cache", exist_ok=True)

# Enable FastF1 cache
fastf1.Cache.enable_cache("./cache")

all_results = []

# Only use 2025 season to avoid API rate limits
years = [2023, 2024, 2025]

for year in years:

    print(f"\nLoading season {year}...")

    try:
        schedule = fastf1.get_event_schedule(year)

        for _, race in schedule.iterrows():

            try:
                print(f"Downloading: {race['EventName']}")

                session = fastf1.get_session(
                    year,
                    race['EventName'],
                    'R'
                )

                # Faster loading, fewer API calls
                session.load(
                    laps=False,
                    telemetry=False,
                    weather=False
                )

                results = session.results.copy()

                results["Year"] = year
                results["Race"] = race["EventName"]

                all_results.append(results)

            except Exception as e:
                print(f"Skipped {race['EventName']}: {e}")

    except Exception as e:
        print(f"Could not load season {year}: {e}")

# Combine all races
if len(all_results) > 0:

    df = pd.concat(all_results, ignore_index=True)

    # Create output folder if needed
    os.makedirs("data", exist_ok=True)

    df.to_csv(
        "data/f1_results.csv",
        index=False
    )

    print("\n✅ Dataset saved successfully!")
    print("Shape:", df.shape)
    print("File: data/f1_results.csv")

else:
    print("\n❌ No data was collected.")