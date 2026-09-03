Data_ID=$1
wandb_key=$2
AWS_S3_PATH=s3://research.luffingfuturellc/Pangolin
#aws s3 cp $AWS_S3_PATH/${Data_ID} ./${Data_ID}


WANDB_API_KEY=${wandb_key} python ./packages/training_lighting.py --input $1 --epochs 200 --model exp --output ${Data_ID/.pt/_model.ckpt}

#python ./packages/training.py --input ./${Data_ID} --epochs 300 --model exp --output ${Data_ID/.pt/_model.pt}

aws s3 cp ${Data_ID/.pt/_model.pt} $AWS_S3_PATH/
