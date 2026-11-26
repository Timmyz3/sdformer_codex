"""Synthetic CPU checks only; no data/checkpoint/GPU or trained-AEE claim."""
import json
import unittest

import torch

from coarse_head import (RuntimeCoarseFlowHead, coarse_supervised_loss,
                         make_tile_targets, shape_operation_counts, targets_as_records)


CONTRACT = {'direction': 'forward_first_to_second', 'time_interval': 'synthetic t0 to t1',
            'units': 'original_image_pixels'}


class RuntimeChecks(unittest.TestCase):
    def test_tail_and_all_T_aggregation_against_explicit_reference(self):
        torch.manual_seed(37)
        u = torch.randn(10, 1, 10, 11, 3, dtype=torch.float64)
        head = RuntimeCoarseFlowHead(3, flow_contract=CONTRACT).double()
        result = head.aggregate(u)
        reference = torch.empty(1, 2, 2, 3, dtype=u.dtype)
        for y in range(2):
            for x in range(2):
                reference[:, y, x] = u[:, :, y*9:min(y*9+9, 10), x*9:min(x*9+9, 11)].mean((0, 2, 3))
        torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)
        # Every T slice contributes, including the final T index.
        u.zero_(); u[-1].fill_(10)
        torch.testing.assert_close(head.aggregate(u), torch.ones_like(result))
        # A tiny actual tail must not be divided by padded 81 points.
        torch.testing.assert_close(head.aggregate(torch.ones(10,1,1,2,3,dtype=u.dtype)),
                                   torch.ones(1,1,1,3,dtype=u.dtype))

    def test_forward_backward_shared_parameters_and_pixel_output(self):
        torch.manual_seed(17)
        u = torch.randn(10,1,10,11,3,requires_grad=True)
        head = RuntimeCoarseFlowHead(3,flow_contract=CONTRACT)
        targets = make_tile_targets(torch.full((2,20,22),7.), torch.ones(20,22,dtype=torch.bool),
                                   feature_hw=(10,11),feature_stride_pixels=(2,2),flow_contract=CONTRACT)
        prediction = head(u)
        self.assertEqual(tuple(prediction.shape),(1,2,2,2))
        report = coarse_supervised_loss(prediction,targets,flow_contract=head.flow_contract,weighting='valid_pixels')
        report['loss'].backward()
        self.assertTrue(torch.isfinite(u.grad).all())
        for parameter in head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(sum(p.numel() for p in head.parameters()),16*3+50)
        # No stride, feature-resolution or tile-size multiplier is hidden in forward.
        with torch.no_grad():
            head.fc2.weight.zero_(); head.fc2.bias.copy_(torch.tensor([12.,-4.]))
        torch.testing.assert_close(head(u), torch.tensor([12.,-4.]).expand(1,2,2,2))

    def test_wrong_T_B_channels_and_contract_rejected(self):
        head=RuntimeCoarseFlowHead(3,flow_contract=CONTRACT)
        for shape in ((9,1,10,10,3),(10,2,10,10,3),(10,1,10,10,4)):
            with self.assertRaises(ValueError):head(torch.zeros(shape))
        with self.assertRaises(ValueError):RuntimeCoarseFlowHead(3,flow_contract={})
        with self.assertRaises(ValueError):RuntimeCoarseFlowHead(3,flow_contract=dict(CONTRACT,units='feature_pixels'))

    def test_declared_operation_counts(self):
        c=shape_operation_counts((10,1,10,11,32))
        self.assertEqual(c['spatial_points_per_tile'],[[81,18],[9,2]])
        self.assertEqual(c['temporal_reduction_additions'],31680)
        self.assertEqual(c['spatial_reduction_additions'],3392)
        self.assertEqual(c['linear_real_MACs'],2176)
        self.assertEqual(c['trainable_parameters'],562)
        self.assertFalse(c['costs_are_measured_cycles'])


class LabelChecks(unittest.TestCase):
    def test_physical_support_valid_only_means_and_real_tail(self):
        gt=torch.empty(2,20,22)
        for y in range(20):
            for x in range(22):gt[:,y,x]=torch.tensor([x*1.,y*1.])
        valid=torch.zeros(20,22,dtype=torch.bool)
        valid[0,0]=True; valid[17,17]=True
        valid[0,18]=True; valid[19,21]=True
        gt[:,1,1]=float('nan')  # Invalid GT is not imputed and does not enter a mean.
        labels=make_tile_targets(gt,valid,feature_hw=(10,11),feature_stride_pixels=(2,2),flow_contract=CONTRACT)
        self.assertEqual(labels['valid_pixel_count'].tolist(),[[[2,1],[0,1]]])
        self.assertEqual(labels['support_pixel_count'].tolist(),[[[324,72],[36,8]]])
        torch.testing.assert_close(labels['flow'][0,0,0],torch.tensor([8.5,8.5]))
        torch.testing.assert_close(labels['flow'][0,0,1],torch.tensor([18.,0.]))
        torch.testing.assert_close(labels['flow'][0,1,1],torch.tensor([21.,19.]))
        self.assertTrue(torch.isnan(labels['flow'][0,1,0]).all())
        records=targets_as_records(labels)
        self.assertIsNone(records[2]['flow_dxdy'])
        self.assertFalse(records[2]['valid'])
        json.dumps(records,allow_nan=False)

    def test_fractional_origin_half_open_support_and_outside(self):
        gt=torch.tensor([[[1.,2.,3.]],[[4.,5.,6.]]])
        valid=torch.ones(1,3,dtype=torch.bool)
        # Tile footprint [0.5,1.5) contains center0.5 only, excludes center1.5.
        labels=make_tile_targets(gt,valid,feature_hw=(1,1),feature_stride_pixels=(1,1),
                                 image_origin_xy=(.5,0),flow_contract=CONTRACT)
        torch.testing.assert_close(labels['flow'][0,0,0],torch.tensor([1.,4.]))
        labels=make_tile_targets(gt,valid,feature_hw=(1,1),feature_stride_pixels=(1,1),
                                 image_origin_xy=(10,10),flow_contract=CONTRACT)
        self.assertEqual(labels['support_pixel_count'].item(),0)
        self.assertFalse(labels['valid'].item())

    def test_valid_nan_bad_mask_and_wrong_direction_rejected(self):
        gt=torch.zeros(2,2,2); valid=torch.ones(2,2,dtype=torch.bool)
        gt[0,0,0]=float('nan')
        with self.assertRaises(ValueError):
            make_tile_targets(gt,valid,feature_hw=(1,1),feature_stride_pixels=(2,2),flow_contract=CONTRACT)
        with self.assertRaises(ValueError):
            make_tile_targets(torch.zeros(2,2,2),torch.full((2,2),.5),feature_hw=(1,1),
                              feature_stride_pixels=(2,2),flow_contract=CONTRACT)
        labels=make_tile_targets(torch.zeros(2,2,2),valid,feature_hw=(1,1),
                                 feature_stride_pixels=(2,2),flow_contract=CONTRACT)
        with self.assertRaises(ValueError):
            coarse_supervised_loss(torch.zeros(1,1,1,2),labels,
                                   flow_contract=dict(CONTRACT,direction='backward_second_to_first'))

    def test_weighting_and_no_GT_skip_without_supervised_zero(self):
        gt=torch.zeros(2,1,10); valid=torch.ones(1,10,dtype=torch.bool)
        labels=make_tile_targets(gt,valid,feature_hw=(1,10),feature_stride_pixels=(1,1),flow_contract=CONTRACT)
        prediction=torch.tensor([[[[1.,1.],[3.,3.]]]],requires_grad=True)
        equal=coarse_supervised_loss(prediction,labels,flow_contract=CONTRACT,weighting='tile')
        weighted=coarse_supervised_loss(prediction,labels,flow_contract=CONTRACT,weighting='valid_pixels')
        self.assertAlmostEqual(equal['loss'].item(),2.)
        self.assertAlmostEqual(weighted['loss'].item(),1.2,places=6)
        self.assertEqual(weighted['weight_sum'],10)
        weighted['loss'].backward()
        torch.testing.assert_close(prediction.grad,torch.tensor([[[[.45,.45],[.05,.05]]]]))
        empty=make_tile_targets(gt,torch.zeros_like(valid),feature_hw=(1,10),
                               feature_stride_pixels=(1,1),flow_contract=CONTRACT)
        report=coarse_supervised_loss(prediction,empty,flow_contract=CONTRACT)
        self.assertIsNone(report['loss']); self.assertTrue(report['skip_optimizer'])
        self.assertTrue(torch.isnan(empty['flow']).all())

    def test_invalid_tile_excluded_before_arithmetic_and_zero_gradient(self):
        gt=torch.zeros(2,1,10); valid=torch.ones(1,10,dtype=torch.bool);valid[0,9]=False
        labels=make_tile_targets(gt,valid,feature_hw=(1,10),feature_stride_pixels=(1,1),flow_contract=CONTRACT)
        prediction=torch.tensor([[[[2.,4.],[7.,9.]]]],requires_grad=True)
        loss=coarse_supervised_loss(prediction,labels,flow_contract=CONTRACT)['loss']
        self.assertEqual(loss.item(),3.)
        loss.backward()
        torch.testing.assert_close(prediction.grad[0,0,1],torch.zeros(2))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
