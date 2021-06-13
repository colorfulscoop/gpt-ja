set -eu

date=${1}
target=jawiki-${date}-pages-articles

# Change directory to open2ch workspace
if [ ! -e jawiki/${date} ]; then
    mkdir -p jawiki/${date}
fi
cd jawiki/${date}

# Downlaod dataset
if [ ! -e ${target}.xml.bz2 ]; then
    wget https://dumps.wikimedia.org/jawiki/${date}/${target}.xml.bz2
fi

# Clone generation repository
if [ ! -e wikiextractor ]; then
    git clone https://github.com/attardi/wikiextractor
    cd wikiextractor
    git checkout 3162bb6c3c9ebd2d15be507aa11d6fa818a454ac
    cd ..
fi

out_dir=out-${date}
if [ ! -e ${out_dir} ]; then
    python wikiextractor/WikiExtractor.py -b 500M --processes 2 --log_file log-${date}.txt -o ${out_dir} ${target}.xml.bz2
    cat ${out_dir}/AA/* >jawiki-${date}-pages-articles.extractor
fi

# Go back to the top directory
cd ../..

target_dir=data/jawiki/${date}
if [ ! -e ${target_dir} ]; then
    echo "Generate train/valid/test dataset in ${target_dir}"
    mkdir -p ${target_dir}/data
    # Generate train/valid/test data
    # [Caution] do not shuffle dataset here for train GPT like dataset.
    cat jawiki/${date}/jawiki-${date}-pages-articles.extractor | grep -v doc | perl -wlp -e 's/。/。\n/g' | perl -wln -e '/^$/ or print' >${target_dir}/all.txt

    head -n500000   ${target_dir}/all.txt                  >${target_dir}/data/valid.txt
    head -n1000000  ${target_dir}/all.txt | tail -n+500001 >${target_dir}/data/test.txt
    tail -n+1000001 ${target_dir}/all.txt                  >${target_dir}/data/train.txt
fi
