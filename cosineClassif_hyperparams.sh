for lr in 1e-5 2e-5; #хочу сюда те, которые вы перебирали; 1e-5 стоял при тестовом запуске по дефолту
do
    for pool in mean; # если это лучший то, только его, у меня при тестовом запуске был first (epochs=2, en-en accuracy = 78), так как default=first. Вопрос добавить ли его
    do
        for add_fc_layer_ in True False; # применяем ли линейные слои на эмбединги слов
        do
	    for emb_size_for_cosine_ in 1024 768;  # output_features  параметр в nn.Linear
	    do
	        for lr_s in constant_warmup linear_warmup; # выбрать лучший, если был определен лучший; constant_warmup при тестовом запуске стоял по дефолту
	        do
                    #OUT_DIR=../../../working/lr-$lr-loss-cosine_similarity-pool-$pool-emb_size_for_cosine-$emb_size_for_cosine_-lrs-$lr_s-add_fc_layer-$add_fc_layer_/
		    OUT_DIR=trained_models/lr-$lr-loss-cosine_similarity-pool-$pool-emb_size_for_cosine-$emb_size_for_cosine_-lrs-$lr_s-add_fc_layer-$add_fc_layer_/
		    python run_model.py --do_train --do_validation \
		        --data_dir ./data/ --output_dir $OUT_DIR \
		        --learning_rate $lr \
		        --loss cosine_similarity \
		        --pool_type $pool \
		        --lr_scheduler $lr_s \
                        --add_fc_layer $add_fc_layer_ \
                        --emb_size_for_cosine $emb_size_for_cosine_ \
                        --model_name xlm-roberta-base;
	        done
	    done
        done
    done
done
