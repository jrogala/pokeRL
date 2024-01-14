for step in range(total_steps):
    latent = world_model.observe(image, latent, action)
    action = actor_critic.act(latent)
    image, reward, done, _ = env.step(action)
    replay_buffer.add(action, image, reward, done)
    world_model.train(replay_buffer)
    actor_critic.train(world_model)