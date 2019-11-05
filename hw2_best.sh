wget 'https://www.dropbox.com/s/0ykbt6cxy932qba/model_bestimproved.pth.tar?dl=1'
mkdir log
mv model_bestimproved.pth.tar?dl=1 ./log/model_bestimproved.pth.tar
python improved_test.py --data_dir $1 --save_data_dir $2