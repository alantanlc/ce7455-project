# WinoGrande 

Version 1.1 (Dec 2nd, 2019)

- - - 

## Data

    ./data/
    ├── train_[xs,s,m,l,xl].jsonl          # training set with differnt sizes
    ├── train_[xs,s,m,l,xl]-labels.lst     # answer labels for training sets
    ├── dev.jsonl                          # development set
    ├── dev-labels.lst                     # answer labels for development set
    ├── test.jsonl                         # test set
    ├── sample-submissions-labels.lst      # example submission file for leaderboard    
    └── eval.py                            # evaluation script
    
You can use `train_*.jsonl` for training models and `dev` for validation.
Please note that labels are not included in `test.jsonl`. To evaluate your models on `test` set, make a submission to our [leaderboard](https://winogrande.allenai.org).

### Training (fine-tuning)

1. You can train your model by doing 'bash sample_training.sh ${MODEL_SIZE}' where MODEL_SIZE can be `xs`, `s`, `m`, `l`, or `xl`.

1. Results will be stored under `./models/${OUTPUT_DIR}`. Please set `${OUTPUT_DIR}` in `sample_training.sh`.

### Prediction (on the test set)

1. You can make predictions by `./scripts/run_experiment.py` directly (see `sample_prediction.sh`).

        e.g., 
        export PYTHONPATH=$PYTHONPATH:$(pwd)

        python scripts/run_experiment.py \
        --model_type roberta_mc \
        --model_name_or_path .output/models \
        --task_name winogrande \
        --do_predict \
        --do_lower_case \
        --data_dir ./data \
        --max_seq_length 80 \
        --per_gpu_eval_batch_size 4 \
        --output_dir ./output/models/ \
        --data_cache_dir ./output/cache/ \

1. If you have an access to [beaker](https://beaker.org/), you can run your experiments  by `sh ./predict_winogrande_on_bkr.sh`.

1. Result is stored in `./output/models/predictions_test.lst`

### Training time on NSCC (Tesla K40)
#### base models
```
xs - 8sec/epoch  
s - 30sec/epoch  
m - 2min/epoch  
l - 8min/epoch  
xl - 30min/epoch  
```
#### large models
```
xs - 1.5min/epoch  
s - 1.5min/epoch  
m - 6min/epoch  
l - 30min/epoch  
xl - 2hrs/epoch  
```
## Evaluation

You can use `eval.py` for evaluation on the dev split, which yields `metrics.json`. 

    e.g., python eval.py --preds_file ./YOUR_PREDICTIONS.lst --labels_file ./dev-labels.lst

In the prediction file, each line consists of the predictions (1 or 2) by 5 training sets (ordered by `xs`, `s`, `m`, `l`, `xl`, separated by comma) for each evauation set question. 

     2,1,1,1,1
     1,1,2,2,2
     1,1,1,1,1
     .........
     .........

Namely, the first column is the predictions by a model trained/finetuned on `train_xs.jsonl`, followed by a model prediction by `train_s.jsonl`, ... , and the last (fifth) column is the predictions by a model from `train_xl.jsonl`.
Please checkout a sample submission file (`sample-submission-labels.lst`) for reference.

## Submission to Leaderboard

You can submit your predictions on `test` set to the [leaderboard](http://winogrande.allenai.org).
The submission file must be named as `predictions.lst`. The format is the same as above.  

    
## Reference
If you use this dataset, please cite the following paper:

	@article{sakaguchi2019winogrande,
	    title={WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
	    author={Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
	    journal={arXiv preprint arXiv:1907.10641},
	    year={2019}
	}


## License 

Winogrande dataset is licensed under CC BY 2.0.


## Questions?

You may ask us questions at our [google group](https://groups.google.com/a/allenai.org/forum/#!forum/winogrande).


## Contact 

Email: keisukes[at]allenai.org
