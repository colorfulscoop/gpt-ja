set -eu

# Args
date=${1}
outdir=${2}

# Target file prefix
save_dir=${outdir}/${date}
target=jawiki-${date}-pages-articles

# Change directory to open2ch workspace
if [ ! -e ${save_dir} ]; then
    mkdir -p ${save_dir}
fi
cd ${save_dir}

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

echo "Generate train/valid/test dataset in ${save_dir}"
# Generate train/valid/test data
# [Caution] do not shuffle dataset here for train GPT like dataset.
cat jawiki-${date}-pages-articles.extractor | \
    grep -v doc | perl -wlp -e 's/。/。\n/g' | \
    perl -wln -e '/^$/ or print' \
    >all.txt

    head -n500000   all.txt                  >valid.txt
    head -n1000000  all.txt | tail -n+500001 >test.txt
    tail -n+1000001 all.txt                  >train.txt
