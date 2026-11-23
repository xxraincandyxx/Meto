import torch.nn.functional as F


def solve(teacher_logits, student_logits, kl, N, V, temperature):
    # Log-softmax uses a row maximum before exponentiation. Keeping the
    # student's probabilities in log space avoids taking log(0) at extremes.
    teacher = F.log_softmax(teacher_logits / temperature, dim=1)
    student = F.log_softmax(student_logits / temperature, dim=1)
    kl.copy_((teacher.exp() * (teacher - student)).sum(dim=1))
