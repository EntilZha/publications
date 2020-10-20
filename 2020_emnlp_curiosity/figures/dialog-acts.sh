 #!/usr/bin/env bash

 pygraph -d -e sfdp 2020_emnlp_curiosity/figures/dialog-acts \
    dialog_acts-inform inform-inform_related inform-inform_unrelated inform-inform_response \
    dialog_acts-offer offer-offer_topic offer-offer_aspect offer-offer_followup offer-offer_accept offer-offer_decline \
    dialog_acts-feedback feedback-feedback_negative feedback-feedback_positive feedback-feedback_elicit \
    dialog_acts-shift_aspect \
    dialog_acts-request request-request_topic request-request_aspect request-request_followup request-request_other
