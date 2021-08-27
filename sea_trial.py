import seaborn as s
flights = s.load_dataset("flights")
print(flights.head())
flights_wide = flights.pivot("year", "month", "passengers")
print(flights_wide.head())
print(flights_wide.values)