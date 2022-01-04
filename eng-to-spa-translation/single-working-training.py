import base

batch_size = 32

train_ds, val_ds = base.eng_to_span_dataset(batch_size)
single_worker_model = base.build_and_compile_transformer_model()
single_worker_model.fit(train_ds, epochs=20, validation_data=val_ds)

single_worker_model.save_weights("weights")
