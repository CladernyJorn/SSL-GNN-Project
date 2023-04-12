try:
    from graphmae import model_graphmae
    from grace import model_grace
    from cca_ssg import model_cca_ssg
    from bgrl import model_bgrl
except:
    from models.graphmae import model_graphmae
    from models.grace import model_grace
    from models.cca_ssg import model_cca_ssg
    from models.bgrl import model_bgrl


def build_model(args):
    if args.model == 'graphmae':
        return model_graphmae(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            nhead_out=args.num_out_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            encoder_type=args.encoder,
            decoder_type=args.decoder,
            mask_rate=args.mask_rate,
            norm=args.norm,
            loss_fn=args.loss_fn,
            drop_edge_rate=args.drop_edge_rate,
            replace_rate=args.replace_rate,
            alpha_l=args.alpha_l,
            concat_hidden=args.concat_hidden,
        )
    elif args.model == 'cca_ssg':
        return model_cca_ssg(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=args.norm,
            encoder_type=args.encoder,
            der=args.der,
            dfr=args.dfr,
            lambd=args.lambd
        )
    elif args.model == 'grace':
        return model_grace(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_proj_hidden=args.num_proj_hidden,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=args.norm,
            encoder_type=args.encoder,
            drop_edge_rate_1=args.drop_edge_rate_1,
            drop_edge_rate_2=args.drop_edge_rate_2,
            drop_feature_rate_1=args.drop_feature_rate_1,
            drop_feature_rate_2=args.drop_feature_rate_2,
            tau=args.tau
        )
    elif args.model == 'bgrl':
        return model_bgrl(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=args.norm,
            encoder_type=args.encoder,
            pred_hid=args.pred_hid,
            moving_average_decay=args.moving_average_decay,
            epochs=args.max_epoch,
            drop_edge_rate_1=args.drop_edge_rate_1,
            drop_edge_rate_2=args.drop_edge_rate_2,
            drop_feature_rate_1=args.drop_feature_rate_1,
            drop_feature_rate_2=args.drop_feature_rate_2,
        )
    else:
        assert False and "Invalid model"