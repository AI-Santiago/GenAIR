import os

from models.Adapter import SASRecPLUS, Bert4RecPLUS, GRU4RecPLUS
from models.utils import FastCalibrationLoss


class GenAIRSASRec(SASRecPLUS):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)

        self.calibration_loss_func = FastCalibrationLoss(
            t=args.tau,
            co_occurrence_file=os.getenv("CO_OCCURRENCE_FILE"),
        )
        self.alpha_calib = args.alpha

        self.filter_init_modules.append("kl_emb")
        self._init_weights()

    def forward(self, seq, pos, neg, positions, **kwargs):
        base_loss = super().forward(seq, pos, neg, positions, **kwargs)

        indices = (pos != 0)
        if indices.any():
            pos_final_embs = self._get_embedding(pos[indices].unsqueeze(1)).squeeze(1)
            calibration_loss = self.calibration_loss_func(pos_final_embs, pos[indices])
            return base_loss + self.alpha_calib * calibration_loss

        return base_loss


class GenAIRBert4Rec(Bert4RecPLUS):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)

        self.calibration_loss_func = FastCalibrationLoss(
            t=args.tau,
            co_occurrence_file=os.getenv("CO_OCCURRENCE_FILE"),
        )
        self.alpha_calib = args.alpha

        self.filter_init_modules.append("kl_emb")
        self._init_weights()

    def forward(self, seq, pos, neg, positions, **kwargs):

        loss = super().forward(seq, pos, neg, positions, **kwargs)

        indices = (pos != 0)
        if indices.any():
            pos_final_embs = self._get_embedding(pos[indices].unsqueeze(1)).squeeze(1)
            calibration_loss = self.calibration_loss_func(pos_final_embs, pos[indices])
            loss = loss + self.alpha_calib * calibration_loss

        return loss


class GenAIRGRU4Rec(GRU4RecPLUS):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)

        self.calibration_loss_func = FastCalibrationLoss(
            t=args.tau,
            co_occurrence_file=os.getenv("CO_OCCURRENCE_FILE"),
        )
        self.alpha_calib = args.alpha

        self.filter_init_modules.append("kl_emb")
        self._init_weights()

    def forward(self, seq, pos, neg, positions, **kwargs):

        loss = super().forward(seq, pos, neg, positions, **kwargs)

        indices = (pos != 0)
        if indices.any():
            pos_final_embs = self._get_embedding(pos[indices].unsqueeze(1)).squeeze(1)
            calibration_loss = self.calibration_loss_func(pos_final_embs, pos[indices])
            loss = loss + self.alpha_calib * calibration_loss

        return loss
