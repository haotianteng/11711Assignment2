export EB_USERNAME="haotiant@andrew.cmu.edu"
export EB_API_KEY="LFo5JxOFlapb2fCo39rvxg"
python -m explainaboard_client.cli.evaluate_system \
  --username $EB_USERNAME \
  --api-key $EB_API_KEY \
  --task named-entity-recognition \
  --system-name anlp_haotiant_sciBertSER \
  --dataset cmu_anlp \
  --sub-dataset sciner \
  --split test \
  --system-output-file test_data/anlp_haotiant_scBertSER.conll \
  --system-output-file-type conll \
  --shared-users neubig@gmail.com \
  --source-language en

