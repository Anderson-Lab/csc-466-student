FOLDER=csc-466-student
rm -rf ../$FOLDER/*
cp -Rp book ../$FOLDER/
cp -Rp labs ../$FOLDER/
cp -Rp config ../$FOLDER/
find ../$FOLDER/ -name "*.ipynb" -exec rm -f {} \;

# TODO go through every md file and remove certain sections
# BEGIN SOLUTION
# END SOLUTION
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

for file in `find ../$FOLDER/book/ -name "*.md" -print`; do
echo $file
filename="${file%.*}"
python - <<EOF
contents = open("$filename.md").read()
lines = []
started_solution = False
for line in contents.split("\n"):
  if "BEGIN SOLUTION" in line:
    started_solution = True
  elif "END SOLUTION" in line:
    started_solution = False
  elif started_solution == False:
    lines.append(line)
open("$filename.md","w").write("\n".join(lines))
EOF
done;

for file in `find ../$FOLDER/labs/ -name "*.md" -print`; do
echo $file
filename="${file%.*}"
python - <<EOF
contents = open("$filename.md").read()
lines = []
started_solution = False
for line in contents.split("\n"):
  if "BEGIN SOLUTION" in line:
    started_solution = True
  elif "END SOLUTION" in line:
    started_solution = False
  elif started_solution == False:
    lines.append(line)
open("$filename.md","w").write("\n".join(lines))
EOF
done;