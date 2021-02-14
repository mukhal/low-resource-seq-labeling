#!/bin/bash

percent=( 10 20 40 80 100)

for i in "${percent[@]}"
do
	cp valid.txt "$i-examples/valid.txt"
	cp test.txt "$i-examples/test.txt"

	cp train.txt "$i-examples/train.txt"
	bash get_last_chunk.sh $i ../egy/train.txt >> "$i-examples/train.txt"
done
