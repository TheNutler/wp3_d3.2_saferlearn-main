#! /bin/bash
OUTPUT="./docker-images"
for img in $(docker-compose config | awk '{if ($1 == "image:") print $2;}'); do
  images="$images $img"
done
images=`echo $images | xargs -n1 | sort -u | xargs`
echo $images
for img in $images; do
    echo "Restoring $img....."  
  gunzip -c $OUTPUT/$img.img.tgz | docker load
done