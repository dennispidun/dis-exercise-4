import base

# This will have to be commented when running the multi-worker

batch_size = 64

single_worker_dataset = eng_to_span_dataset(batch_size)
single_worker_model = build_and_compile_transformer_model()
# single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)
single_worker_model.fit(single_worker_dataset, epochs=3, )
