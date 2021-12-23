import base

# This will have to be commented when running the multi-worker

batch_size = 64

single_worker_dataset = base.eng_to_span_dataset(batch_size)
single_worker_model = base.build_and_compile_transformer_model()
# single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)
single_worker_model.fit(
    single_worker_dataset,
    epochs=10,
)

single_worker_model.save_weights("weights")
