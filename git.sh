YEAR=$(date +%Y)
MONTH=$(date +%m)
DAY=$(date +%d)

git add .
git commit -m [$YEAR.$MONTH.$DAY]
git push -u origin master