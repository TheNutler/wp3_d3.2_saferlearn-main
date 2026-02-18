#! /bin/bash
OUTPUT="./docker-images"
for img in $(docker-compose config | awk '{if ($1 == "image:") print $2;}'); do
    images="$images $img"
done
images=`echo $images | xargs -n1 | sort -u | xargs`
echo $images
for img in $images; do
    img=$(echo $img | cut -d '/' -f 1)
    echo "Saving $img....."
    # docker save $img | gzip > $img.img.tgz
    docker save $img | tqdm --bytes --total $(docker image inspect $img --format='{{.Size}}') > $OUTPUT/$img.tar
    echo "Compressing"
    gzip $OUTPUT/$img.tar
done
