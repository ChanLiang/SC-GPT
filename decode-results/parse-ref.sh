fi=$1
fo=$2
cat $fi | cut -f 2 -d '&' | sed 's/^ *//g' > fo
