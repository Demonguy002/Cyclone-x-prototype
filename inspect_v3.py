import joblib

MODEL_PATH = r"E:\Cyclone-x-prototype\model\cyclonex_v3_stacking_model.joblib"
SCALER_PATH = r"E:\Cyclone-x-prototype\model\cyclonex_v3_scaler.joblib"

print("=" * 75)
print("CYCLONE-X V3 — COMPLETE MODEL INTERFACE INSPECTION")
print("=" * 75)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("\nMODEL")
print("Type:", type(model).__name__)

print("\nSCALER")
print("Type:", type(scaler).__name__)
print("Input features:", getattr(scaler, "n_features_in_", "UNKNOWN"))

print("\nMODEL INPUT FEATURES")
print(getattr(model, "n_features_in_", "UNKNOWN"))

print("\nCLASSES")
print(getattr(model, "classes_", "UNKNOWN"))

print("\nSCALER FEATURE NAMES")
if hasattr(scaler, "feature_names_in_"):
    for i, name in enumerate(scaler.feature_names_in_, 1):
        print(f"{i:03d}. {name}")
else:
    print("No feature_names_in_ stored in scaler.")

print("\nMODEL FEATURE NAMES")
if hasattr(model, "feature_names_in_"):
    for i, name in enumerate(model.feature_names_in_, 1):
        print(f"{i:03d}. {name}")
else:
    print("No feature_names_in_ stored in stacking model.")

print("\nBASE ESTIMATORS")
if hasattr(model, "estimators"):
    for name, estimator in model.estimators:
        print(f"{name:15s} -> {type(estimator).__name__}")
        print("   features:",
              getattr(estimator, "n_features_in_", "UNKNOWN"))

print("\nFINAL ESTIMATOR")
if hasattr(model, "final_estimator_"):
    final_model = model.final_estimator_
    print("Type:", type(final_model).__name__)
    print("Features:",
          getattr(final_model, "n_features_in_", "UNKNOWN"))

print("\nSCALER PARAMETERS")
print("Means:", len(getattr(scaler, "mean_", [])))
print("Scales:", len(getattr(scaler, "scale_", [])))

print("\nFIRST 10 SCALER MEANS")
if hasattr(scaler, "mean_"):
    print(scaler.mean_[:10])

print("\nFIRST 10 SCALER SCALES")
if hasattr(scaler, "scale_"):
    print(scaler.scale_[:10])

print("\n" + "=" * 75)
print("INSPECTION COMPLETE")
print("=" * 75)
