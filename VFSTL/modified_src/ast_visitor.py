import math
import operator
import collections

from rtamt.syntax.ast.visitor.stl.ast_visitor import StlAstVisitor
from rtamt.semantics.enumerations.comp_oper import StlComparisonOperator
from rtamt.exception.exception import RTAMTException
# yiting
import torch as th
import time
class StlDiscreteTimeOfflineAstVisitor(StlAstVisitor):

    def visit(self, node, *args, **kwargs):
        result = super(StlDiscreteTimeOfflineAstVisitor, self).visit(node, *args, **kwargs)
        self.ast.offline_results[node] = result
        return result

    def visitPredicate(self, node, *args, **kwargs):
        start_time = time.time()
        sample_left  = self.visit(node.children[0], *args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        # print("s0000L time: {}".format(total_time))
        sample_right = self.visit(node.children[1], *args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        # print("s0000L time: {}".format(total_time))
        # sample_right = th.tensor(sample_right, device=sample_left.device)
        end_time = time.time()
        total_time = end_time - start_time
        # print(sample_right.device)
        # print(sample_left.device)
        # print("s0000 sample time: {}".format(total_time))
    
        sample_return = th.empty(sample_left.size(), device=sample_left.device)
        if node.operator.value == StlComparisonOperator.EQ.value:
            sample_return = -th.abs(sample_left - sample_right)
        elif node.operator.value == StlComparisonOperator.NEQ.value:
            sample_return = th.abs(sample_left - sample_right)
        elif node.operator.value == StlComparisonOperator.LEQ.value or node.operator.value == StlComparisonOperator.LESS.value:
            sample_return = sample_right - sample_left
        elif node.operator.value == StlComparisonOperator.GEQ.value or node.operator.value == StlComparisonOperator.GREATER.value:
            sample_return = sample_left - sample_right
        else:   
            raise RTAMTException('Unknown predicate operation')
        end_time = time.time()
        total_time = end_time - start_time
        # print("s0000 minus time: {}".format(total_time))
        return sample_return
        
        # for i in range(len(sample_left)):
        #     if node.operator.value == StlComparisonOperator.EQ.value:
        #         val = -abs(sample_left[i] - sample_right[i])
        #     elif node.operator.value == StlComparisonOperator.NEQ.value:
        #         val = abs(sample_left[i] - sample_right[i])
        #     elif node.operator.value == StlComparisonOperator.LEQ.value or node.operator.value == StlComparisonOperator.LESS.value:
        #         val = sample_right[i] - sample_left[i]
        #     elif node.operator.value == StlComparisonOperator.GEQ.value or node.operator.value == StlComparisonOperator.GREATER.value:
        #         val = sample_left[i] - sample_right[i]
        #     else:
        #         raise RTAMTException('Unknown predicate operation')
        #     sample_return.append(val)
        # return sample_return


    def visitVariable(self, node, *args, **kwargs):
        var = self.ast.var_object_dict[node.var]
        if node.field:  #TODO Tom did not understand this line.
            sample_return = []
            for v in var:
                sample_return.append(operator.attrgetter(node.field)(v))
        else:
            sample_return = var
        return sample_return


    def visitAbs(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        sample_return = []
        for i in sample:
            out_sample = abs(i)
            sample_return.append(out_sample)
        return sample_return

    def visitSqrt(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        sample_return = []
        for i in sample:
            out_sample = math.sqrt(i)
            sample_return.append(out_sample)
        return sample_return

    def visitExp(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        sample_return = []
        for i in sample:
            out_sample = math.exp(i)
            sample_return.append(out_sample)
        return sample_return

    def visitPow(self, node, *args, **kwargs):
        sample_1 = self.visit(node.children[0], *args, **kwargs)
        sample_2 = self.visit(node.children[1], *args, **kwargs)

        sample_return = []
        for i in range(len(sample_1)):
            out_sample = math.pow(sample_1[i], sample_2[i])
            sample_return.append(out_sample)
        return sample_return


    def visitAddition(self, node, *args, **kwargs):
        sample_left  = self.visit(node.children[0], *args, **kwargs)
        sample_right = self.visit(node.children[1], *args, **kwargs)

        sample_return = []
        for i in range(len(sample_left)):
            out_sample = sample_left[i] + sample_right[i]
            sample_return.append(out_sample)
        return sample_return


    def visitSubtraction(self, node, *args, **kwargs):
        sample_left  = self.visit(node.children[0], *args, **kwargs)
        sample_right = self.visit(node.children[1], *args, **kwargs)

        sample_return = []
        for i in range(len(sample_left)):
            out_sample = sample_left[i] - sample_right[i]
            sample_return.append(out_sample)
        return sample_return


    def visitMultiplication(self, node, *args, **kwargs):
        sample_left  = self.visit(node.children[0], *args, **kwargs)
        sample_right = self.visit(node.children[1], *args, **kwargs)

        sample_return = []
        for i in range(len(sample_left)):
            out_sample = sample_left[i] * sample_right[i]
            sample_return.append(out_sample)
        return sample_return


    def visitDivision(self, node, *args, **kwargs):
        sample_left  = self.visit(node.children[0], *args, **kwargs)
        sample_right = self.visit(node.children[1], *args, **kwargs)

        sample_return = []
        for i in range(len(sample_left)):
            out_sample = sample_left[i] / sample_right[i]
            sample_return.append(out_sample)
        return sample_return


    def visitNot(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)
        sample = -sample 
        #sample_return = [ -i for i in sample]
        return sample


    def visitAnd(self, node, *args, **kwargs):
        # sample_left  = self.visit(node.children[0], *args, **kwargs)
        # sample_right = self.visit(node.children[1], *args, **kwargs)
        #print('start and')
        # sample_left  = th.vstack(self.visit(node.children[0], *args, **kwargs))
        # sample_right = th.vstack(self.visit(node.children[1], *args, **kwargs))
        # #print('end and')
        # sample_return = th.minimum(sample_left, sample_right)
        
        sample_left  = self.visit(node.children[0], *args, **kwargs)
        sample_right = self.visit(node.children[1], *args, **kwargs)
        #print('end and')
        sample_return = th.min(sample_left, sample_right)
        
        #sample_return = list(map(min, zip(sample_left, sample_right)))
        return sample_return


    def visitOr(self, node, *args, **kwargs):
        # sample_left  = self.visit(node.children[0], *args, **kwargs)
        # sample_right = self.visit(node.children[1], *args, **kwargs)

        # sample_return = list(map(max, zip(sample_left, sample_right)))
        # return sample_return
        sample_left  = th.vstack(self.visit(node.children[0], *args, **kwargs))
        sample_right = th.vstack(self.visit(node.children[1], *args, **kwargs))
        #print('end and')
        sample_return = th.maximum(sample_left, sample_right)
        
        #sample_return = list(map(min, zip(sample_left, sample_right)))
        return [e for e in sample_return]


    def visitImplies(self, node, *args, **kwargs):
        sample_left  = self.visit(node.children[0], *args, **kwargs)
        sample_right = self.visit(node.children[1], *args, **kwargs)

        sample_return = [max(-l,r) for l,r in zip(sample_left, sample_right)]
        return sample_return


    def visitIff(self, node, *args, **kwargs):
        sample_left  = self.visit(node.children[0], *args, **kwargs)
        sample_right = self.visit(node.children[1], *args, **kwargs)

        sample_return = [-abs(l-r) for l,r in zip(sample_left, sample_right)]
        return sample_return


    def visitXor(self, node, *args, **kwargs):
        sample_left  = self.visit(node.children[0], *args, **kwargs)
        sample_right = self.visit(node.children[1], *args, **kwargs)

        sample_return = [abs(l-r) for l,r in zip(sample_left, sample_right)]
        return sample_return


    def visitEventually(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        sample_return = []
        prev_out = -float("inf")
        for i in reversed(sample):
            out_sample = max(i, prev_out)
            prev_out = out_sample
            sample_return.append(out_sample)
        sample_return.reverse()
        return sample_return


    def visitAlways(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        sample_return = []
        prev_out = float("inf")
        for i in reversed(sample):
            out_sample = min(i, prev_out)
            prev_out = out_sample
            sample_return.append(out_sample)
        sample_return.reverse()
        return sample_return


    def visitUntil(self, node, *args, **kwargs):
        sample_left  = self.visit(node.children[0], *args, **kwargs)
        sample_right = self.visit(node.children[1], *args, **kwargs)

        sample_return = []
        next_out = -float("inf")
        for i in range(len(sample_left)-1, -1, -1):
            out_sample = min(sample_left[i], next_out)
            out_sample = max(out_sample, sample_right[i])
            next_out = out_sample
            sample_return.append(out_sample)
        sample_return.reverse()
        return sample_return


    def visitOnce(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        sample_return = []
        prev_out = -float("inf")
        for i in sample:
            out_sample = max(i, prev_out)
            prev_out = out_sample
            sample_return.append(out_sample)
        return sample_return


    def visitHistorically(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        sample_return = []
        prev_out = float("inf")
        for i in sample:
            out_sample = min(i, prev_out)
            prev_out = out_sample
            sample_return.append(out_sample)
        return sample_return


    def visitSince(self, node, *args, **kwargs):
        sample_left  = self.visit(node.children[0], *args, **kwargs)
        sample_right = self.visit(node.children[1], *args, **kwargs)

        sample_return = []
        prev_out = -float("inf")
        for i in range(len(sample_left)):
            out_sample = min(sample_left[i], prev_out)
            out_sample = max(out_sample, sample_right[i])
            prev_out = out_sample
            sample_return.append(out_sample)
        return sample_return


    def visitRise(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        prev = sample[:-1]
        prev.insert(0,-float("inf"))
        sample_return = [min(-p,s) for p,s in zip(prev, sample)]
        return sample_return


    def visitFall(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        prev = sample[:-1]
        prev.insert(0,float("inf"))
        sample_return = [min(p,-s) for p,s in zip(prev, sample)]
        return sample_return


    def visitConstant(self, node, *args, **kwargs):
        length = args[0]
        # return [node.val]*length
        return th.full((length,), node.val, device=th.device("cuda")).view(-1, 1)


    def visitPrevious(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        sample_return = []
        prev = float("inf")
        for i in sample:
            out_sample = prev
            prev = i
            sample_return.append(out_sample)
        return sample_return

    def visitStrongPrevious(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        sample_return = []
        prev = -float("inf")
        for i in sample:
            out_sample = prev
            prev = i
            sample_return.append(out_sample)
        return sample_return


    def visitNext(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        sample_return = sample[1:]
        sample_return.append(float("inf"))
        return sample_return

    def visitStrongNext(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)

        sample_return = sample[1:]
        sample_return.append(-float("inf"))
        return sample_return


    def visitTimedPrecedes(self, node, *args, **kwargs):
        raise RTAMTException('Offline does not need visitTimedPrecedes')


    def visitTimedOnce(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)
        begin, end = self.time_unit_transformer(node)

        sample = [-float("inf") for j in range(end)] + sample
        sample_return = [max(sample[j - end:j - begin+ 1]) for j in range(end, len(sample))]
        return sample_return


    def visitTimedHistorically(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)
        begin, end = self.time_unit_transformer(node)

        sample = [float("inf") for j in range(end)] + sample
        sample_return = [min(sample[j - end:j - begin + 1]) for j in range(end, len(sample))]
        return sample_return

    def visitTimedSince(self, node, *args, **kwargs):
        sample_left  = self.visit(node.children[0], *args, **kwargs)
        sample_right = self.visit(node.children[1], *args, **kwargs)
        begin, end = self.time_unit_transformer(node)

        sample_return = []
        buffer_left = collections.deque(maxlen=(end + 1))
        buffer_right = collections.deque(maxlen=(end + 1))

        for i in range(end + 1):
            s_left = float("inf")
            s_right = - float("inf")
            buffer_left.append(s_left)
            buffer_right.append(s_right)

        for i in range(len(sample_left)):
            buffer_left.append(sample_left[i])
            buffer_right.append(sample_right[i])
            out_sample = - float("inf")
            

            for j in range(end-begin+1):
                c_left = float("inf")
                c_right = buffer_right[j]
                for k in range(j+1, end+1):
                    c_left = min(c_left, buffer_left[k])
                out_sample = max(out_sample, min(c_left, c_right))
            sample_return.append(out_sample)
        return sample_return


    def visitTimedAlways(self, node, *args, **kwargs):
        sample = self.visit(node.children[0], *args, **kwargs)
        begin, end = self.time_unit_transformer(node)
        sample_len = len(sample)
        if sample_len <= end:
            sample = th.vstack([sample, th.ones(end - sample_len + 1, sample.size()[1], device=sample.device)*float('inf') ])
        device = sample.device
        # diff = end - begin
        # sample_return  = [min(sample[j:j+diff+1]) for j in range(begin, end+1)]
        # tmp  = [min(sample[j:j+diff+1]) for j in range(end+1,len(sample))]
        # sample_return += tmp
        # tmp  = [float("inf") for j in range(len(sample)-len(sample_return))]
        # sample_return += tmp
        diff = end - begin
        horizon = diff + 1
        batch_size = sample.size()[1]
        vectorized_sample = th.cat((sample, self.infTensorMatrix((horizon-1, batch_size), device)), dim=0).unfold(0, horizon, 1)
        sample_return = th.min(vectorized_sample, dim=2).values
        return sample_return[0:sample_len]


    
    def visitTimedEventually(self, node, *args, **kwargs):
        # sample = self.visit(node.children[0], *args, **kwargs)
        # begin, end = self.time_unit_transformer(node)
        # sample_len = len(sample)

        # if sample_len <= end:
        #     sample += [-float('inf')] * (end - sample_len + 1)

        # diff = end - begin
        # sample_return  = [max(sample[j:j+diff+1]) for j in range(begin, end+1)]
        # tmp = [max(sample[j:j+diff+1]) for j in range(end+1,len(sample))]
        # sample_return += tmp
        # tmp = [-float("inf") for j in range(len(sample)-len(sample_return))]
        # sample_return += tmp
        # return sample_return[0:sample_len]
        sample = self.visit(node.children[0], *args, **kwargs)
        device = sample.device
        begin, end = self.time_unit_transformer(node)
        sample_len = len(sample)
        #print(begin, end)
        if sample_len <= end:
            sample = th.vstack([sample, th.ones(end - sample_len + 1, sample.size()[1], device=sample.device)*float('inf')*-1 ])
            #sample += [-float('inf')] * (end - sample_len + 1)
        # diff = end - begin
        # sample_return  = [th.max(sample[j:j+diff+1], 0).values for j in range(begin, end+1)]
        # tmp = [th.max(sample[j:j+diff+1], 0).values for j in range(end+1,len(sample))]
        # sample_return += tmp
        # tmp = [-float("inf") for j in range(len(sample)-len(sample_return))]
        # sample_return += tmp
        diff = end - begin
        horizon = diff + 1
        batch_size = sample.size()[1]
        vectorized_sample = th.cat((sample, -1*self.infTensorMatrix((horizon-1, batch_size), device)), dim=0).unfold(0, horizon, 1)
        sample_return = th.max(vectorized_sample, dim=2).values
        
        return sample_return[0:sample_len]

    # def infTensorMatrix(self, size, device):
    #     return th.ones(size, device=device)*float('inf')
    
    # def minTwoTensor(self, t1, t2):
    #     return th.min(th.vstack([t1, t2]), 0).values
    
    # def maxTwoTensor(self, t1, t2):
    #     return th.max(th.vstack([t1, t2]), 0).values
        
    def batched_tensor_push_to_queue(self, tensor_queue: th.Tensor, tensor_element:th.Tensor):
        return th.cat((tensor_queue[1:], tensor_element.view(1, -1))).to(tensor_queue.device)
    
    # def visitTimedUntil(self, node, *args, **kwargs):

    #     sample_left  = self.visit(node.children[0], *args, **kwargs)
    #     sample_right = self.visit(node.children[1], *args, **kwargs)
    #     device = sample_left[0].device
    #     batch_size = sample_left[0].size()[0]
    #     begin, end = self.time_unit_transformer(node)
    #     sample_return = []
    #     queue_len = end + 1
    #     buffer_left = self.infTensorMatrix((queue_len, batch_size), device)
    #     buffer_right = -1 * self.infTensorMatrix((queue_len, batch_size), device)
    #     for i in range(len(sample_left)-1, -1, -1):
    #         buffer_left = self.tensor_push_to_queue(buffer_left, sample_left[i])
    #         buffer_right =self.tensor_push_to_queue(buffer_right, sample_right[i])
    #         out_sample = -1 * self.infTensorMatrix(batch_size, device)
    #         for j in range(end-begin+1):
    #             c_right = buffer_right[j]
    #             c_left = None
    #             if(j < end-begin+1 - 1):
    #                 c_left = th.min(buffer_left[j+1:end+1, :], 0).values
    #             else:
    #                 c_left = c_right
    #             out_sample = self.maxTwoTensor(out_sample, self.minTwoTensor(c_left, c_right))
    #         sample_return.append(out_sample)
    #     sample_return.reverse()
        
    #     return sample_return

    def infTensorMatrix(self, size, device):
        return th.full(size, float('inf'), device=device)

    def minTwoTensor(self, t1, t2):
        # Assuming t1 and t2 can be batches of tensors
        return th.min(t1, t2)

    def maxTwoTensor(self, t1, t2):
        # Assuming t1 and t2 can be batches of tensors
        return th.max(t1, t2)

    # def batched_tensor_push_to_queue(self, tensor_queue, tensor_elements):
    #     # Assumes tensor_elements is batched along the first dimension
    #     return th.cat((tensor_queue[:, 1:, :], tensor_elements.unsqueeze(1)), dim=1)

    
    def visitTimedUntil(self, node, *args, **kwargs):
        
        # import time
        # start_time = time.time()

        # sample_left = self.visit(node.children[0], *args, **kwargs)  # Assuming these return batched samples
        # sample_right = self.visit(node.children[1], *args, **kwargs)
        # device = sample_left[0].device
        # batch_size = sample_left[0].size()[0]
        # begin, end = self.time_unit_transformer(node)
        # # Initialize buffers
        # num_samples = len(sample_left)
        # queue_len = end + 1
        # buffer_left = self.infTensorMatrix((num_samples, queue_len, batch_size), device)
        # buffer_right = -1 * self.infTensorMatrix((num_samples, queue_len, batch_size), device)

        # # sample_return = []
        # # #sample_left : T * N
        # # buffer_left = collections.deque(maxlen=(end + 1))
        # # buffer_right = collections.deque(maxlen=(end + 1))

        # # for i in range(end + 1):
        # #     s_left = self.infTensorMatrix((batch_size,), device)
        # #     s_right = -1 * self.infTensorMatrix((batch_size,), device)
        # #     buffer_left.append(s_left)
        # #     buffer_right.append(s_right)
        # # for i in range(len(sample_left)-1, -1, -1):
        # #     buffer_left.append(sample_left[i])
        # #     buffer_right.append(sample_right[i])
        # #     out_sample = -1 * self.infTensorMatrix(batch_size, device)
        # #     print(i)
        # #     for j in range(end-begin+1):
        # #         c_left = self.infTensorMatrix(batch_size, device)
        # #         c_right = buffer_right[j]
        # #         for k in range(j+1, end+1): # next moment of c right
        # #             # c_left = th.min(th.vstack([c_left, buffer_left[k]]), 0)
        # #             c_left = self.minTwoTensor(c_left, buffer_left[k])
        # #         out_sample = self.maxTwoTensor(out_sample, self.minTwoTensor(c_left, c_right))
        # #     sample_return.append(out_sample)
        # # sample_return.reverse()
        # # test_sample_return = sample_return
        
        # # Convert sample_left and sample_right to tensors for batch processing
        # sample_left_tensor = sample_left.flip(0)
        # sample_right_tensor = sample_right.flip(0)
        # # Update buffers in a batched manner
        # for i in range(num_samples):
        #     buffer_left[i] = self.batched_tensor_push_to_queue(buffer_left[i-1] if i > 0 else buffer_left[i], sample_left_tensor[i])
        #     buffer_right[i] = self.batched_tensor_push_to_queue(buffer_right[i-1] if i > 0 else buffer_right[i], sample_right_tensor[i])
        # # Prepare output samples
        # sample_return_M = th.empty((num_samples, batch_size), device=device)

        # # Batch process the merging of left and right samples
        # for i in range(num_samples):
        #     out_sample = -1 * self.infTensorMatrix((batch_size,), device)
        #     for j in range(end - begin + 1):
        #         c_right = buffer_right[i, j]
        #         c_left = th.min(buffer_left[i, (j + 1):(end + 1)], dim=0).values if j < end - begin else c_right
        #         out_sample = self.maxTwoTensor(out_sample, self.minTwoTensor(c_left, c_right))
        #     sample_return_M[i] = out_sample
        # # Reverse the batched samples to match original order
        # sample_return_M = sample_return_M.flip(0)
        # # return sample_return
        # # return [sample_return[i] for i in range(sample_return.size(0))]
        # old = sample_return_M
        # end_time = time.time()
        # total_time = end_time - start_time
        # print("!!!original Time {}".format(total_time))
        
        start_time = time.time()
        
        sample_left = self.visit(node.children[0], *args, **kwargs)  # Assuming these return batched samples
        sample_right = self.visit(node.children[1], *args, **kwargs)
        device = sample_left[0].device
        batch_size = sample_left[0].size()[0]
        begin, end = self.time_unit_transformer(node)

        # Initialize buffers
        num_samples = len(sample_left)
        queue_len = end + 1
        horizon = queue_len - begin
        # buffer_left = self.infTensorMatrix((num_samples, queue_len, batch_size), device)
        # buffer_right = -1 * self.infTensorMatrix((num_samples, queue_len, batch_size), device)

        # # Convert sample_left and sample_right to tensors for batch processing
        # sample_left_tensor = th.stack(sample_left[::-1])  # Reverse to match original logic
        # sample_right_tensor = th.stack(sample_right[::-1])
        sample_left_tensor = sample_left.to(th.float).flip(0)
        sample_right_tensor = sample_right.to(th.float).flip(0)

        # No more loops: using advanced indexing to update all buffers simultaneously
        # tmp_left = th.cat(( float('inf')*th.ones((queue_len-1, )), batch_size), dim=0)
        # tmp_left = 
        # tmp_right = 
        horizon = end - begin + 1
        buffer_left = th.cat( (self.infTensorMatrix((end-1, batch_size), device), sample_left_tensor), dim=0).unfold(0, end, 1)
        buffer_right = th.cat( ( -self.infTensorMatrix((end-1, batch_size), device), sample_right_tensor), dim=0).unfold(0, end, 1)
        
        # left until right
        buffer_left = th.cat((buffer_left, float('inf')*th.ones((buffer_left.size()[0], buffer_left.size()[1], horizon-1), device=device)), dim=2)
        buffer_left = buffer_left.unfold(2, horizon-1, 1)
        buffer_left = th.min(buffer_left, dim=3).values[:,:,1:]
        sample_return = th.max(th.min(buffer_left, buffer_right), dim=2).values
        sample_return = sample_return.flip(0)
        
        
        # #indices = th.arange(num_samples, device=device)
        
        # # buffer_left[indices] = th.cat((buffer_left[:, 1:, :], th.stack(sample_left_tensor.unsqueeze(1)) ), dim=1)
        # # buffer_right[indices] = th.cat((buffer_right[:, 1:, :], sample_right_tensor.unsqueeze(1)), dim=1)
        
        
        

        # # # Now handle computation of out_samples without looping through samples or queue positions
        # # # Extend dimensions for broadcasting where necessary and use advanced indexing
        # # expanded_right = buffer_right[:, begin:end + 1, :].unsqueeze(2)
        # # expanded_left = th.min(buffer_left[:, begin + 1:end + 2, :].unsqueeze(2), dim=1).values
        # # Assuming buffer_left and buffer_right are [batch, time, features]
        # expanded_right = buffer_right[:, begin:end + 1, :].unsqueeze(2)  # This becomes [batch, time, 1, features]

        # # For expanded_left, we want to maintain the batch and feature dimensions while collapsing the time dimension
        # # First, ensure that we apply min across the correct dimension and then unsqueeze to match expanded_right
        # # The unsqueeze(2) here is to match the [batch, time, 1, features] format of expanded_right
        # min_values, _ = th.min(buffer_left[:, begin + 1:end + 2, :], dim=1, keepdim=True)  # [batch, 1, features]
        # expanded_left = min_values.unsqueeze(2)  # [batch, 1, 1, features]
        # # Min and Max operations across all samples and positions, leveraging broadcasting
        # out_samples = self.maxTwoTensor(expanded_left, expanded_right)
        # out_samples_min_across_time = th.min(out_samples, dim=1).values

        # end_time = time.time()
        # total_time = end_time - start_time
        # print("!!!vecterized {}".format(total_time))
        # # No need to reverse order in vectorized form, adjust according to how you process outputs
        # return out_samples_min_across_time
        
        # new_end_time = time.time()
        # total_time = new_end_time - start_time
        # print("origin time: {}".format(total_time))
        # return sample_return
        
        return sample_return
        # # old = sample_return
        # end_time = time.time()
        # total_time = end_time - start_time
        # print("optimzed time: {}".format(total_time))
        # start_time = time.time()
        # sample_left  = self.visit(node.children[0], *args, **kwargs)
        # sample_right = self.visit(node.children[1], *args, **kwargs)
        # device = sample_left[0].device
        # batch_size = sample_left[0].size()[0]
        # begin, end = self.time_unit_transformer(node)
        # sample_return = []
        # queue_len = end + 1
        # buffer_left = self.infTensorMatrix((queue_len, batch_size), device)
        # buffer_right = -1 * self.infTensorMatrix((queue_len, batch_size), device)
        # new_end_time = time.time()
        # total_time = new_end_time - start_time
        # print("brfore origion: {}".format(total_time))
        
        # start_time = time.time()
        # for i in range(len(sample_left)-1, -1, -1):
        #     print(i)
        #     buffer_left = self.batched_tensor_push_to_queue(buffer_left, sample_left[i])
        #     buffer_right =self.batched_tensor_push_to_queue(buffer_right, sample_right[i])
        #     out_sample = -1 * self.infTensorMatrix((batch_size,), device)
        #     for j in range(end-begin+1):
        #         c_right = buffer_right[j]
        #         c_left = None
        #         if(j < end-begin+1 - 1):
        #             c_left = th.min(buffer_left[j+1:end+1, :], 0).values
        #         else:
        #             c_left = c_right
        #         out_sample = self.maxTwoTensor(out_sample, self.minTwoTensor(c_left, c_right))
        #     sample_return.append(out_sample)
        # sample_return.reverse()
        
        # new_end_time = time.time()
        # total_time = new_end_time - start_time
        # print("origin time: {}".format(total_time))
        # return sample_return
        
    
        # sample_left  = self.visit(node.children[0], *args, **kwargs)
        # sample_right = self.visit(node.children[1], *args, **kwargs)
        # begin, end = self.time_unit_transformer(node)

        # sample_return = []
        # buffer_left = collections.deque(maxlen=(end + 1))
        # buffer_right = collections.deque(maxlen=(end + 1))

        # for i in range(end + 1):
        #     s_left = float("inf")
        #     s_right = - float("inf")
        #     buffer_left.append(s_left)
        #     buffer_right.append(s_right)
        # for i in range(len(sample_left)-1, -1, -1):
        #     buffer_left.append(sample_left[i])
        #     buffer_right.append(sample_right[i])
        #     out_sample = - float("inf")

        #     for j in range(end-begin+1):
        #         c_left = float("inf")
        #         c_right = buffer_right[j]
        #         for k in range(j+1, end+1):
        #             c_left = min(c_left, buffer_left[k])
        #         out_sample = max(out_sample, min(c_left, c_right))
        #     sample_return.append(out_sample)
        # sample_return.reverse()
        # return sample_return
        
        # sample_left : T * N
            # buffer_left = collections.deque(maxlen=(end + 1))
        # buffer_right = collections.deque(maxlen=(end + 1))

        # for i in range(end + 1):
        #     s_left = self.infTensorMatrix(batch_size, device)
        #     s_right = -1 * self.infTensorMatrix(batch_size, device)
        #     buffer_left.append(s_left)
        #     buffer_right.append(s_right)
        # for i in range(len(sample_left)-1, -1, -1):
        #     buffer_left.append(sample_left[i])
        #     buffer_right.append(sample_right[i])
        #     out_sample = -1 * self.infTensorMatrix(batch_size, device)
        #     print(i)
        #     for j in range(end-begin+1):
        #         c_left = self.infTensorMatrix(batch_size, device)
        #         c_right = buffer_right[j]
        #         for k in range(j+1, end+1): # next moment of c right
        #             # c_left = th.min(th.vstack([c_left, buffer_left[k]]), 0)
        #             c_left = self.minTwoTensor(c_left, buffer_left[k])
        #         out_sample = self.maxTwoTensor(out_sample, self.minTwoTensor(c_left, c_right))
        #     sample_return.append(out_sample)
        # sample_return.reverse()
        # test_sample_return = sample_return
