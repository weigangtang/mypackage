from twilio.rest import Client

#------------------------------------------------------------------------------------
# Account Info for w29tang@hotmail.com 
# account_sid = 'ACb362c96bddef365a148f15282e99fda2'
# auth_token  = '2b59a6225d3dc7dedeab8e36ceff073b'
# from_number = '+12892046132'

#------------------------------------------------------------------------------------
# Account Info for w29tang@gmail.com 
account_sid = 'AC45f3644d372453e8a8f2e7ba3452ebed'
auth_token  = '80a57f043426c0abcf520f43087fcdea'
from_number = '+13312085765'

# FUNC ------------------------------------------------------------------------------
def send_text(msg_to_send, to_number):
	client = Client(account_sid, auth_token)
	msg = client.messages.create(body=msg_to_send, from_=from_number, to=to_number)
	print(msg.sid)
