wget 'https://www.dropbox.com/s/6517561r9n7fat7/model_best.pth.tar?dl=1'
mkdir log
mv model_best.pth.tar?dl=1 ./log/model_best.pth.tar
python test.py --data_dir $1 --save_data_dir $2