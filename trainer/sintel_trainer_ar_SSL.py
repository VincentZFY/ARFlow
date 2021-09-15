import time
import torch
from torchvision import transforms
from copy import deepcopy
from .base_trainer_SSL import BaseTrainerSSL
from utils.flow_utils import evaluate_flow, supervised_loss,flow_to_image
from utils.misc_utils import AverageMeter
from transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from transforms.ar_transforms.oc_transforms import run_slic_pt, random_crop


class TrainFramework(BaseTrainerSSL):
    def __init__(self, supervised_loader, unsupervised_loader, valid_loader, model, loss_func, supervised_loss_func,
                 _log, save_root, config):
        super(TrainFramework, self).__init__(
            supervised_loader, unsupervised_loader, valid_loader, model, loss_func, supervised_loss_func,_log, save_root, config)

        self.sp_transform = RandomAffineFlow(
            self.cfg.st_cfg, addnoise=self.cfg.st_cfg.add_noise,crop=self.cfg.st_cfg.crop).to(self.device)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['loss','l_supervised','EPE_0', 'EPE_1', 'EPE_2', 'EPE_3','l_weak','l_unsupervised', 'l_ph', 'l_sm', 'flow_mean']
        # key_meter_names = ['loss','l_supervised','EPE_0', 'EPE_1', 'EPE_2', 'EPE_3']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        self.model.train()
        end = time.time()
        if 'stage1' in self.cfg:
            if self.i_epoch == self.cfg.stage1.epoch:
                self.loss_func.cfg.update(self.cfg.stage1.loss)
        unsupervised_iter=iter(self.unsupervised_loader)

        for i_step, data in enumerate(self.supervised_loader):
            if i_step > self.cfg.epoch_size:
                break
            # read data to device
            # img1, img2 = data['img1'].to(self.device), data['img2'].to(self.device)
            # img_pair = torch.cat([img1, img2], 1)
            # gt_flows = data['target']['flow'].permute(0, 2, 3, 1)

            # img_flow = flow_to_image(data['target']['flow'][0].numpy().transpose([1, 2, 0]))
            # img_flow = transforms.ToPILImage()(img_flow)
            # img_flow.save("./save_flow.jpg")
            # save_img1 = data['img1_ph'][0].cpu().clone()
            # save_img1 = save_img1.squeeze(0)
            # save_img1 = transforms.ToPILImage()(save_img1)
            # save_img1.save("./save_img1.jpg")
            # save_img2 = data['img2_ph'][0].cpu().clone()
            # save_img2 = save_img2.squeeze(0)
            # save_img2 = transforms.ToPILImage()(save_img2)
            # save_img2.save("./save_img2.jpg")

            s_supervised = {'imgs': [data['img1_ph'].to(self.device),
                              data['img2_ph'].to(self.device)],
                     'flows_f': [data['target']['flow'].to(self.device)]}
            s_supervised = self.sp_transform(s_supervised)
            
            # sp_img1 = s_supervised['imgs'][0][0].cpu().clone()
            # sp_img1 = sp_img1.squeeze(0)
            # sp_img1 = transforms.ToPILImage()(sp_img1)
            # sp_img1.save('./sp_img1.jpg')
            # sp_img2 = s_supervised['imgs'][1][0].cpu().clone()
            # sp_img2 = sp_img2.squeeze(0)
            # sp_img2 = transforms.ToPILImage()(sp_img2)
            # sp_img2.save('./sp_img2.jpg')
            # sp_flow = flow_to_image(s_supervised['flows_f'][0][0].cpu().numpy().transpose([1, 2, 0]))
            # sp_flow = transforms.ToPILImage()(sp_flow)
            # sp_flow.save('./sp_flow.jpg')

            img_pair = torch.cat(s_supervised['imgs'], 1).to(self.device)
            flow_gt = s_supervised['flows_f'][0].to(self.device)

            # measure data loading time
            am_data_time.update(time.time() - end)
            loss = 0.0
            flows_12 = self.model(img_pair, with_bk=False)['flows_fw']
            # pred_flows = flow_to_image(flows_12[0][0].detach().cpu().numpy().transpose([1, 2, 0]))
            # pred_flows = transforms.ToPILImage()(pred_flows)
            # pred_flows.save('./pred_flow.jpg')

            l_supervised, pyramid_epe = self.supervised_loss_func(flows_12, flow_gt)
            loss += l_supervised / 10000
            
            try:
                datau = unsupervised_iter.next()
            except:
                unsupervised_iter = iter(self.unsupervised_loader)
                datau = unsupervised_iter.next()
            imgu1, imgu2 = datau['img1_ph'].to(self.device), datau['img2_ph'].to(self.device)
            imgu1_og, imgu2_og = datau['img1'].to(self.device), datau['img2'].to(self.device)
            imgu_pair = torch.cat([imgu1, imgu2], 1)
            imgu_pair_og = torch.cat([imgu1_og, imgu2_og], 1)

            resu_dict_og = self.model(imgu_pair_og, with_bk=True)
            flowsu_12_og, flowsu_21_og = resu_dict_og['flows_fw'], resu_dict_og['flows_bw']
            flowsu_og = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                     zip(flowsu_12_og, flowsu_21_og)]

            l_weak, l_ph, l_sm, flow_mean = self.loss_func(flowsu_og, imgu_pair_og)

            # # resu_dict = self.model(imgu_pair, with_bk=True)
            # # flowsu_12, flowsu_21 = resu_dict['flows_fw'], resu_dict['flows_bw']
            # # flowsu = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
            # #          zip(flowsu_12, flowsu_21)]
            loss += l_weak

            flowsu_weak = resu_dict_og['flows_fw'][0].detach()
  
            noc_ori = self.loss_func.pyramid_occu_mask1[0]  # non-occluded region
            s = {'imgs': [imgu1, imgu2], 'flows_f': [flowsu_weak], 'masks_f': [noc_ori]}
            st_res = self.sp_transform(deepcopy(s))
            flow_t, noc_t = st_res['flows_f'][0], st_res['masks_f'][0]

            # run 2nd pass
            img_pair = torch.cat(st_res['imgs'], 1)
            flow_t_pred = self.model(img_pair, with_bk=False)['flows_fw'][0]

            if not self.cfg.mask_st:
                noc_t = torch.ones_like(noc_t)
            l_unsupervised = ((flow_t_pred - flow_t).abs() + self.cfg.ar_eps) ** self.cfg.ar_q
            l_unsupervised = (l_unsupervised * noc_t).mean() / (noc_t.mean() + 1e-7)

            loss += l_unsupervised

            # update meters
            key_meters.update(
                [loss.item(), l_supervised.item(), pyramid_epe[0].item(), pyramid_epe[1].item(),
                pyramid_epe[2].item(),pyramid_epe[3].item(),l_weak.item(),
                l_unsupervised.item(), l_ph.item(), l_sm.item(), flow_mean.item()],
                img_pair.size(0))
            # key_meters.update(
            #     [loss.item(), l_supervised.item(), pyramid_epe[0].item(), pyramid_epe[1].item(),
            #      pyramid_epe[2].item(),pyramid_epe[3].item()],
            #     img_pair.size(0))


            # compute gradient and do optimization step
            self.optimizer.zero_grad()
            # loss.backward()

            scaled_loss = 1024. * loss
            scaled_loss.backward()

            for param in [p for p in self.model.parameters() if p.requires_grad]:
                param.grad.data.mul_(1. / 1024)

            self.optimizer.step()

            # measure elapsed time
            am_batch_time.update(time.time() - end)
            end = time.time()

            if self.i_iter % self.cfg.record_freq == 0:
                for v, name in zip(key_meters.val, key_meter_names):
                    self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)

            if self.i_iter % self.cfg.print_freq == 0:
                istr = '{}:{:04d}/{:04d}'.format(
                    self.i_epoch, i_step, self.cfg.epoch_size) + \
                       ' Time {} Data {}'.format(am_batch_time, am_data_time) + \
                       ' Info {}'.format(key_meters)
                self._log.info(istr)

            self.i_iter += 1
        self.i_epoch += 1

    @torch.no_grad()
    def _validate_with_gt(self):
        batch_time = AverageMeter()

        if type(self.valid_loader) is not list:
            self.valid_loader = [self.valid_loader]

        # only use the first GPU to run validation, multiple GPUs might raise error.
        # https://github.com/Eromera/erfnet_pytorch/issues/2#issuecomment-486142360
        self.model = self.model.module
        self.model.eval()

        end = time.time()

        all_error_names = []
        all_error_avgs = []

        n_step = 0
        for i_set, loader in enumerate(self.valid_loader):
            error_names = ['EPE']
            error_meters = AverageMeter(i=len(error_names))
            for i_step, data in enumerate(loader):
                img1, img2 = data['img1'], data['img2']
                img_pair = torch.cat([img1, img2], 1).to(self.device)
                gt_flows = data['target']['flow'].numpy().transpose([0, 2, 3, 1])

                # compute output
                flows = self.model(img_pair)['flows_fw']
                pred_flows = flows[0].detach().cpu().numpy().transpose([0, 2, 3, 1])

                es = evaluate_flow(gt_flows, pred_flows)
                error_meters.update([l.item() for l in es], img_pair.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i_step % self.cfg.print_freq == 0 or i_step == len(loader) - 1:
                    self._log.info('Test: {0}[{1}/{2}]\t Time {3}\t '.format(
                        i_set, i_step, self.cfg.valid_size, batch_time) + ' '.join(
                        map('{:.2f}'.format, error_meters.avg)))

                if i_step > self.cfg.valid_size:
                    break
            n_step += len(loader)

            # write error to tf board.
            for value, name in zip(error_meters.avg, error_names):
                self.summary_writer.add_scalar(
                    'Valid_{}_{}'.format(name, i_set), value, self.i_epoch)

            all_error_avgs.extend(error_meters.avg)
            all_error_names.extend(['{}_{}'.format(name, i_set) for name in error_names])

        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.cfg.save_iter:
            print(self.i_epoch)
            self.save_model(all_error_avgs[0] + all_error_avgs[1], name='Sintel')

        return all_error_avgs, all_error_names
