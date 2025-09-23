import subprocess

print("🚀 Running generate_dataset.py ...")
result = subprocess.run(["python", "generate_dataset.py"])

if result.returncode == 0:
    print("✅ Done. Running refresh_all.py ...")
    subprocess.run(["python", "refresh_all.py"])
else:
    print("❌ generate_dataset.py failed. Skipping refresh_all.py.")

print("🎯 All done.")
