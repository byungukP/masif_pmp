masif_root=/masif    # root_dir manually defined based on last WORKDIR in Dockerfile
#masif_root=$(git rev-parse --show-toplevel)      # causing ownership error: "fatal: detected dubious ownership in repository at '/masif' To add an exception for this directory, call: git config --global --add safe.directory /masif"
                                                  # when tried the command --> error: could not lock config file //.gitconfig: Permission denied
masif_source=$masif_root/source/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_site/

echo "Training Start:"
date
python3 $masif_source/masif_site/masif_site_train.py $1
echo "Training Done:"
date
