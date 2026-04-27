# DEST_DIR=./data
# mkdir -p $DEST_DIR
# URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
# ZIP_FILE=./data/afhq.zip
# mkdir -p ./data
# wget -N $URL -O $ZIP_FILE
# unzip $ZIP_FILE -d $DEST_DIR
# rm $ZIP_FILE
# mv ./data/afhq ./data/afhq_cat
# mv ./data/afhq_cat/val ./data/afhq_cat/test
# bash scripts/afhq_validation_images.sh

# DEST_DIR=./data/celeba
# ZIP_FILE="$DEST_DIR/celeba-dataset.zip"
# mkdir -p $DEST_DIR
# echo "Downloading CelebA dataset..."
# kaggle datasets download jessicali9530/celeba-dataset -p "$DEST_DIR"
# # Ensure the ZIP file exists before extracting
# if [ -f "$ZIP_FILE" ]; then
#     echo "Dataset downloaded. Extracting..."
#     unzip -q "$ZIP_FILE" -d "$DEST_DIR"
#     rm "$ZIP_FILE"
#     echo "Extraction completed!"
# else
#     echo "Error: ZIP file not found after download!"
#     exit 1
# mv ./data/celeba/img_align_celeba/img_align_celeba/* ./data/celeba/img_align_celeba
# fi

DEST_DIR=./data
GOPRO_DEST="$DEST_DIR/gopro"
GOPRO_TMP="$DEST_DIR/gopro_tmp_extract"
GOPRO_ZIP="$DEST_DIR/GOPRO_Large.zip"
GOPRO_URL="https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip?download=true"

if [ "${DOWNLOAD_GOPRO:-1}" = "1" ]; then
    if [ -d "$GOPRO_DEST/train" ] && [ -d "$GOPRO_DEST/test" ]; then
        echo "GoPro dataset already exists at $GOPRO_DEST. Skipping download."
    else
        echo "Downloading GoPro GOPRO_Large dataset..."
        mkdir -p "$DEST_DIR"
        wget -N "$GOPRO_URL" -O "$GOPRO_ZIP"

        if [ -f "$GOPRO_ZIP" ]; then
            echo "GoPro dataset downloaded. Extracting..."
            rm -rf "$GOPRO_TMP"
            mkdir -p "$GOPRO_TMP"
            unzip -q "$GOPRO_ZIP" -d "$GOPRO_TMP"
            rm "$GOPRO_ZIP"

            rm -rf "$GOPRO_DEST"
            if [ -d "$GOPRO_TMP/GOPRO_Large" ]; then
                mv "$GOPRO_TMP/GOPRO_Large" "$GOPRO_DEST"
            elif [ -d "$GOPRO_TMP/train" ] && [ -d "$GOPRO_TMP/test" ]; then
                mv "$GOPRO_TMP" "$GOPRO_DEST"
                GOPRO_TMP=""
            else
                echo "Error: unexpected GoPro archive layout."
                exit 1
            fi

            if [ -n "$GOPRO_TMP" ]; then
                rm -rf "$GOPRO_TMP"
            fi
            echo "GoPro extraction completed at $GOPRO_DEST."
        else
            echo "Error: GoPro ZIP file not found after download!"
            exit 1
        fi
    fi
else
    echo "Skipping GoPro download because DOWNLOAD_GOPRO=$DOWNLOAD_GOPRO."
fi
