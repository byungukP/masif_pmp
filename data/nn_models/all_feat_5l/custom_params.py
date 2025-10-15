custom_params = {}
custom_params["range_val_samples"] = 0.1
custom_params['model_dir'] = 'nn_models/all_feat_5l/model_data/'
custom_params['feat_mask'] = [1.0, 1.0, 1.0, 1.0, 1.0]  # shape index, ddc, hbond, pb, hphob
custom_params['n_conv_layers'] = 5
custom_params['out_pred_dir'] = 'output/all_feat_5l/pred_data/'
custom_params['out_surf_dir'] = 'output/all_feat_5l/pred_surfaces/'
custom_params['epoch_num'] = 100
